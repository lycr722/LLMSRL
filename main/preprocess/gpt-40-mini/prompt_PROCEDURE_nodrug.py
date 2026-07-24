import pandas as pd
import requests
import time
import csv
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================================
# API 与并发设置
# ================================
API_URL = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}

MAX_RETRY = 3  # 局部机制：单次请求遇到网络错误时的立即重试次数
RETRY_DELAY = 5  # 局部机制：重试前的休眠时间（秒）
MAX_WORKERS = 10  # 并发线程数
MAX_SWEEPS = 5  # 全局机制：最终检查的“最大扫尾轮数”

# 线程锁，确保多线程安全
csv_write_lock = threading.Lock()
log_write_lock = threading.Lock()


# ================================
# Constrained Diagnosis Prompt
# ================================
def get_diagnosis_description(icd9_code):
    prompt = (
        "You are a senior clinician-scientist and biomedical informatician. Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. "
        "The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the required therapeutic mechanisms. "
        "For ICD-9-CM diagnosis codes\"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow. "
        "Embed relevant disease/procedure codes naturally within the text immediately after the term, using the format 'ICD-9-CM: [code]'.\n"
        "CRITICAL CONSTRAINT: You must NOT mention any specific drug names, active ingredients, or medication codes (such as ATC/NDC). Instead, describe the broader therapeutic intent or physiological mechanisms targeted.\n"
        "1. Core Definition: Briefly define the diagnosis and its typical clinical impact. "
        "2. Common co-existing diseases (with their ICD-9-CM diagnosis codes) that influence treatment strategies. "
        "3. Major Contraindications: Describe the pharmacological mechanisms or general therapeutic effects that should be avoided or used with extreme caution, and explain why. "
        "4. Relevant medical or surgical procedures (with ICD-9-CM procedure codes) commonly associated with this diagnosis. "
        "5. Treatment goals and therapeutic intent: What are the primary treatment goals? Describe the physiological pathways or biological mechanisms that typical treatments aim to modulate.\n"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.1,
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ================================
# Constrained Procedure Prompt
# ================================
def get_procedure_description(icd9_code):
    prompt = (
        "You are an expert clinical informatician and surgical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code. "
        "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent physiological needs. "
        "For the ICD-9-CM procedure code: \"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. Embed all disease/procedure codes naturally using 'ICD-9-CM: [code]'.\n"
        "CRITICAL CONSTRAINT: You must NOT mention any specific drug names, active ingredients, or medication codes (such as ATC/NDC). Instead, describe the broader therapeutic intent or recovery requirements.\n"
        "1. Procedure Definition & Purpose.\n"
        "2. Primary Indications (ICD-9-CM diagnoses).\n"
        "3. Pre-procedure physiological preparations: Describe the physiological states or biological adjustments required before surgery (e.g., managing coagulation or reducing inflammation).\n"
        "4. Post-procedure recovery needs: Describe the physiological requirements for postoperative care (e.g., pain management mechanisms, infection prevention strategies).\n"
        "5. Key contraindications.\n"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.1,
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ================================
# 自动重试包装器 (局部)
# ================================
def call_with_retry(api_func, code, fail_log_file):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return api_func(code)
        except Exception as e:
            # 遇到报错，短暂休眠后再次尝试
            time.sleep(RETRY_DELAY)

    # 局部重试全部失败，返回错误标识（不写入 CSV，留给全局扫尾去处理）
    with log_write_lock:
        with open(fail_log_file, "a") as f:
            f.write(f"{code} failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    return "ERROR"


# ================================
# 主处理函数 (包含全局扫尾与最终检查)
# ================================
def process_codes(code_type='diagnosis'):
    # 1. 基础配置
    if code_type == 'diagnosis':
        vocab_path = "official_vocabs.pkl"
        if not os.path.exists(vocab_path):
            print("❌ official_vocabs.pkl not found.")
            return
        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        icd9_codes_all = sorted(set(vocabs["diagnoses"]))

        output_csv = "DIAGNOSES_Description_Constrained_ID.csv"
        api_function = get_diagnosis_description
        fail_log = "FAILED_DIAG_CODES.txt"
        print("🚀 Task: Processing DIAGNOSIS codes (Constrained)")

    elif code_type == 'procedure':
        input_csv = "mimic3_procedures_mapping.csv"
        df = pd.read_csv(input_csv, dtype={"ICD9_CODE": str})
        icd9_codes_all = sorted(set(df["ICD9_CODE"].dropna()))

        output_csv = "PROCEDURES_Description_Constrained_ID.csv"
        api_function = get_procedure_description
        fail_log = "FAILED_PROC_CODES.txt"
        print("🚀 Task: Processing PROCEDURE codes (Constrained)")
    else:
        print("❌ Invalid code_type")
        return

    # 初始化 CSV 表头
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ICD9_CODE", "DESCRIPTION"])

    def process_single(code):
        desc = call_with_retry(api_function, code, fail_log)
        return code, desc

    # 2. 全局扫尾与最终检查循环
    sweep_round = 1
    while sweep_round <= MAX_SWEEPS:
        # --- 最终检查：从 CSV 中读取真正成功的记录 ---
        completed = set()
        try:
            df_exist = pd.read_csv(output_csv, dtype={"ICD9_CODE": str})
            # 过滤掉空值或意外写入的 ERROR，确保提取的都是合法的特征
            valid_df = df_exist[df_exist["DESCRIPTION"].notna() & (~df_exist["DESCRIPTION"].str.startswith("ERROR"))]
            completed = set(valid_df["ICD9_CODE"])
        except Exception as e:
            pass

        # 找出还需要跑的代码
        pending = [c for c in icd9_codes_all if c not in completed]

        if not pending:
            print("🎉 Final Check Passed! 100% of the codes are successfully processed!")
            return

        print(
            f"\n🔄 [Sweep Round {sweep_round}/{MAX_SWEEPS}] Total: {len(icd9_codes_all)} | Completed: {len(completed)} | Pending: {len(pending)}")

        # --- 多线程执行当前待处理的任务 ---
        with open(output_csv, "a", newline='', encoding="utf-8") as f:
            csv_writer = csv.writer(f)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_code = {executor.submit(process_single, code): code for code in pending}

                for future in tqdm(as_completed(future_to_code), total=len(pending)):
                    code, desc = future.result()

                    # 只有真正成功的生成才会被写入 CSV
                    if desc != "ERROR":
                        with csv_write_lock:
                            csv_writer.writerow([code, desc])
                            f.flush()

        sweep_round += 1

        # 如果还要进行下一轮扫尾，先让程序休息 10 秒（防止被 API 持续封禁）
        if sweep_round <= MAX_SWEEPS and pending:
            print("⏳ Round finished. Waiting 10 seconds before the next verification sweep...")
            time.sleep(10)

    print(f"⚠️ Reached max sweeps ({MAX_SWEEPS}). Some codes might still be missing. Please check {fail_log}.")


# ================================
# 运行入口
# ================================
if __name__ == "__main__":
    # 推荐分别运行，以获得最清晰的日志和控制
    # process_codes("diagnosis")
    process_codes("procedure")