import pandas as pd
import requests
import time
import csv
import os
import pickle
from tqdm import tqdm

# ================================
# API 设置
# ================================
API_URL = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}

MAX_RETRY = 3          # 每个 code 最多重试次数
RETRY_DELAY = 3        # 重试前等待秒数
REQUEST_DELAY = 2      # 正常请求间隔


# ================================
# Diagnosis Prompt
# ================================
def get_diagnosis_description(icd9_code):
    prompt = (
        "You are a senior clinician-scientist and biomedical informatician. Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. "
        "The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
        "For ICD-9-CM diagnosis codes\"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow. "
        "Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Core Definition: Briefly define the diagnosis and its typical clinical impact."
        "2. Common co-existing diseases (with their ICD-9-CM diagnosis codes) that influence polypharmacy. "
        "3. Major Contraindications: Specify drugs or drug classes (with ATC-3 codes) that should be avoided or used with extreme caution, and explain why. "
        "4. Relevant medical or surgical procedures (with ICD-9-CM procedure codes) commonly associated with this diagnosis. "
        "5. Treatment goals and typical drug classes: What are the primary treatment goals? What first-line and second-line drug classes (with ATC-3 codes) are typically prescribed?\n"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ================================
# Procedure Prompt
# ================================
def get_procedure_description(icd9_code):
    prompt = (
        "You are an expert clinical informatician and surgical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code. "
        "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
        "For the ICD-9-CM procedure code: \"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text. Never use quotation marks for any terms, names, or codes. "
        "Do not use bullet points, headings, or numbered lists. Embed all medical codes naturally using 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Procedure Definition & Purpose.\n"
        "2. Primary Indications (ICD-9-CM diagnoses).\n"
        "3. Pre-procedure medication adjustments (ATC-3 codes).\n"
        "4. Post-procedure medications (ATC-3 codes).\n"
        "5. Key contraindications.\n"
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,
        "stream": False
    }

    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ================================
# 自动重试包装器
# ================================
def call_with_retry(api_func, code, fail_log_file):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return api_func(code)
        except Exception as e:
            print(f"⚠️ Error for {code} (attempt {attempt}/{MAX_RETRY}): {e}")
            time.sleep(RETRY_DELAY)

    # 全部失败 → 写入失败日志
    with open(fail_log_file, "a") as f:
        f.write(code + "\n")

    return f"ERROR: Failed after {MAX_RETRY} attempts"


# ================================
# 主处理函数
# ================================
def process_codes(code_type='diagnosis'):
    if code_type == 'diagnosis':
        vocab_path = "official_vocabs.pkl"
        if not os.path.exists(vocab_path):
            print("❌ official_vocabs.pkl not found.")
            return

        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)

        icd9_codes_all = sorted(set(vocabs["diagnoses"]))
        output_csv = "DIAGNOSES_Description_ID.csv"
        api_function = get_diagnosis_description
        fail_log = "FAILED_DIAG_CODES.txt"

        print("🚀 Processing DIAGNOSIS codes...")

    elif code_type == 'procedure':
        input_csv = "mimic3_procedures_mapping.csv"
        df = pd.read_csv(input_csv, dtype={"ICD9_CODE": str})
        icd9_codes_all = sorted(set(df["ICD9_CODE"].dropna()))

        output_csv = "PROCEDURES_Description_ID.csv"
        api_function = get_procedure_description
        fail_log = "FAILED_PROC_CODES.txt"

        print("🚀 Processing PROCEDURE codes...")

    else:
        print("❌ Invalid code_type")
        return

    # 已完成的代码
    completed = set()
    if os.path.exists(output_csv):
        try:
            df_exist = pd.read_csv(output_csv, dtype={"ICD9_CODE": str})
            completed = set(df_exist["ICD9_CODE"])
        except:
            pass

    pending = [c for c in icd9_codes_all if c not in completed]
    print(f"Total: {len(icd9_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")

    # 初始化输出文件
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ICD9_CODE", "DESCRIPTION"])

    # 逐条处理
    with open(output_csv, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)

        for code in tqdm(pending):
            desc = call_with_retry(api_function, code, fail_log)
            writer.writerow([code, desc])
            f.flush()
            time.sleep(REQUEST_DELAY)

    print("🎉 All done!")


# ================================
# 运行
# ================================
if __name__ == "__main__":
    # process_codes("diagnosis")
    process_codes("procedure")
