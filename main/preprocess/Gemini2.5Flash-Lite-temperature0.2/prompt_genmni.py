import os
import csv
import time
import pickle
import requests
from tqdm import tqdm

# =========================================================
# 基础配置
# C:\ProgramData\anaconda3\envs\env_lamrec\python.exe D:\Code\LAMRec-RAGBK\preprocess\Gemini2.5Flash-Lite\prompt.py
# ====================================================
# Starting generation pipeline with Gemini 2.5 Flash-Lite
# ====================================================
#
# 🚀 Processing DRUG codes...
# Total: 131, Completed: 131, Pending: 0
# ✅ Drug descriptions saved to 'MIMIC3_Drug_Descriptions_ID.csv'
#
# 🚀 Processing DIAGNOSIS codes...
# Generating drug descriptions: 0it [00:00, ?it/s]
# Generating diagnosis descriptions:   0%|          | 0/100 [00:00<?, ?it/s]Total: 1958, Completed: 1858, Pending: 100
# Generating diagnosis descriptions: 100%|██████████| 100/100 [10:41<00:00,  6.41s/it]
# Generating procedure descriptions:   0%|          | 0/1430 [00:00<?, ?it/s]✅ Diagnosis descriptions saved to 'DIAGNOSES_Description_ID.csv'
#
# 🚀 Processing PROCEDURE codes...
# Total: 1430, Completed: 0, Pending: 1430
# Generating procedure descriptions:  14%|█▍        | 204/1430 [22:16<1:52:52,  5.52s/it]⚠️ Error for 2411 (attempt 1/3): HTTPSConnectionPool(host='api.zhizengzeng.com', port=443): Max retries exceeded with url: /google/v1beta/models/gemini-2.5-flash-lite:generateContent?key=sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x000002907E8D46D0>: Failed to establish a new connection: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。'))
# Generating procedure descriptions:  28%|██▊       | 404/1430 [46:53<8:54:07, 31.24s/it] ⚠️ Error for 3801 (attempt 1/3): HTTPSConnectionPool(host='api.zhizengzeng.com', port=443): Read timed out. (read timeout=90)
# Generating procedure descriptions:  38%|███▊      | 548/1430 [1:10:29<1:23:40,  5.69s/it]⚠️ Error for 4389 (attempt 1/3): HTTPSConnectionPool(host='api.zhizengzeng.com', port=443): Max retries exceeded with url: /google/v1beta/models/gemini-2.5-flash-lite:generateContent?key=sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x000002907E9059D0>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
# Generating procedure descriptions:  53%|█████▎    | 760/1430 [1:37:15<1:10:01,  6.27s/it]⚠️ Error for 5310 (attempt 1/3): HTTPSConnectionPool(host='api.zhizengzeng.com', port=443): Max retries exceeded with url: /google/v1beta/models/gemini-2.5-flash-lite:generateContent?key=sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x000002907E905A60>: Failed to establish a new connection: [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。'))
# Generating procedure descriptions:  55%|█████▌    | 788/1430 [1:41:14<2:11:01, 12.25s/it]⚠️ Error for 544 (attempt 1/3): HTTPSConnectionPool(host='api.zhizengzeng.com', port=443): Max retries exceeded with url: /google/v1beta/models/gemini-2.5-flash-lite:generateContent?key=sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x000002907E905C70>: Failed to establish a new connection: [Errno 11001] getaddrinfo failed'))
# Generating procedure descriptions: 100%|██████████| 1430/1430 [2:45:56<00:00,  6.96s/it]
# ✅ Procedure descriptions saved to 'PROCEDURES_Description_ID.csv'
#
# 🎉 All tasks completed!
#
# Process finished with exit code 0

# =========================================================
GEMINI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"
MODEL_NAME = "gemini-2.5-flash-lite"
API_URL = f"https://api.zhizengzeng.com/google/v1beta/models/{MODEL_NAME}:generateContent"

HEADERS = {
    "Content-Type": "application/json"
}

REQUEST_DELAY = 2
MAX_RETRY = 3
RETRY_DELAY = 3
TIMEOUT = 90

# =========================================================
# 输入输出文件
# =========================================================
DRUG_INPUT_CSV = "drug_code2index.csv"
NDC_ATC_MAP_CSV = "ndc_atc_map_level3.csv"
DRUG_OUTPUT_CSV = "MIMIC3_Drug_Descriptions_ID.csv"
DRUG_FAIL_LOG = "FAILED_DRUG_CODES.txt"

VOCAB_PKL = "official_vocabs.pkl"
DIAG_OUTPUT_CSV = "DIAGNOSES_Description_ID.csv"
DIAG_FAIL_LOG = "FAILED_DIAG_CODES.txt"

PROC_INPUT_CSV = "mimic3_procedures_mapping.csv"
PROC_OUTPUT_CSV = "PROCEDURES_Description_ID.csv"
PROC_FAIL_LOG = "FAILED_PROC_CODES.txt"

# =========================================================
# 系统指令
# =========================================================
SYSTEM_INSTRUCTION = (
    "You are a precise clinical biomedical writing assistant. "
    "Follow the user instruction exactly. "
    "Generate formal, clinically grounded, machine-readable medical text. "
    "Do not use bullet points, headings, numbered lists, or quotation marks unless explicitly requested. "
    "Keep the output as one continuous paragraph."
)

# =========================================================
# Prompt 模板
# =========================================================
DRUG_PROMPT_TMPL = (
    "You are a distinguished clinical pharmacologist and expert in cheminformatics. "
    "Your objective is to generate a comprehensive, machine-readable description for a drug class, "
    "identified by its ATC-3 code. This description must bridge its clinical application with its "
    "underlying pharmacological properties, enabling an AI model to learn complex drug-drug, "
    "drug-procedure, and drug-disease relationships. "
    "For the medication with ATC level-3 code: {atc3_code} "
    "(representative NDC: {ndc_code}).\n\n"
    "Synthesize the following information into a single continuous paragraph of formal medical text. "
    "Never use quotation marks for any terms, names, or codes. "
    "Do not use bullet points, headings, or numbered lists. "
    "If a specific point is not clinically relevant or applicable, omit it to maintain a natural flow. "
    "Embed all medical codes naturally within the text immediately after the relevant term, using the formats "
    "ICD-9-CM: [code] and ATC-3: [code].\n"
    "1. State Anatomical Main Group (ATC Level 1): State the drug's broader anatomical main group.\n"
    "2. Define Therapeutic Context (ATC Level 2): Describe the therapeutic subgroup. "
    "List the typical diseases it treats with ICD-9-CM diagnosis codes and associated clinical procedures "
    "with ICD-9-CM procedure codes.\n"
    "3. Detail Pharmacological Profile and Interactions (ATC Levels 3-5): "
    "Detail the specific pharmacological class, its precise mechanism of action, common synergistic "
    "co-prescriptions with ATC-3 codes, and critical drug-drug interactions.\n"
)

DIAGNOSIS_PROMPT_TMPL = (
    "You are a senior clinician-scientist and biomedical informatician. "
    "Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. "
    "The goal is to generate a description that helps a machine learning model understand the impact of "
    "the diagnosis on the recommended drug combination. "
    "For ICD-9-CM diagnosis code: {icd9_code}\n\n"
    "Synthesize the following information into a single continuous paragraph of formal medical text. "
    "Never use quotation marks for any terms, names, or codes. "
    "Do not use bullet points, headings, or numbered lists. "
    "If a specific point is not clinically relevant or applicable, omit it from the description to maintain a natural flow. "
    "Embed all medical codes naturally within the text immediately after the relevant term, using the formats "
    "ICD-9-CM: [code] and ATC-3: [code].\n"
    "1. Core Definition: Briefly define the diagnosis and its typical clinical impact.\n"
    "2. Common co-existing diseases with their ICD-9-CM diagnosis codes that influence polypharmacy.\n"
    "3. Major Contraindications: Specify drugs or drug classes with ATC-3 codes that should be avoided "
    "or used with extreme caution, and explain why.\n"
    "4. Relevant medical or surgical procedures with ICD-9-CM procedure codes commonly associated with this diagnosis.\n"
    "5. Treatment goals and typical drug classes: What are the primary treatment goals? "
    "What first-line and second-line drug classes with ATC-3 codes are typically prescribed?\n"
)

PROCEDURE_PROMPT_TMPL = (
    "You are an expert clinical informatician and surgical pharmacologist. "
    "Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code. "
    "The goal is to generate a description that helps a machine learning model understand the context and purpose "
    "of this procedure, especially its relationship to subsequent medication needs. "
    "For the ICD-9-CM procedure code: {icd9_code}\n\n"
    "Synthesize the following information into a single continuous paragraph of formal medical text. "
    "Never use quotation marks for any terms, names, or codes. "
    "Do not use bullet points, headings, or numbered lists. "
    "Embed all medical codes naturally using ICD-9-CM: [code] and ATC-3: [code].\n"
    "1. Procedure definition and purpose.\n"
    "2. Primary indications with ICD-9-CM diagnosis codes.\n"
    "3. Pre-procedure medication adjustments with ATC-3 codes.\n"
    "4. Post-procedure medications with ATC-3 codes.\n"
    "5. Key contraindications.\n"
)

# =========================================================
# 通用工具函数
# =========================================================
def check_api_key():
    if not GEMINI_API_KEY.strip():
        raise ValueError("GEMINI_API_KEY 为空，请直接在代码里填入你的 key。")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())

def append_fail_code(fail_log_file: str, code: str):
    with open(fail_log_file, "a", encoding="utf-8") as f:
        f.write(str(code) + "\n")

def ensure_csv_header(output_csv: str, header: list):
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def read_completed_codes(output_csv: str, key_col: str) -> set:
    completed = set()
    if not os.path.exists(output_csv):
        return completed

    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get(key_col, "")
                if value is not None:
                    value = str(value).strip()
                    if value:
                        completed.add(value)
    except Exception as e:
        print(f"⚠️ 读取已完成文件失败 {output_csv}: {e}")

    return completed

def read_csv_column(file_path: str, column_name: str) -> list:
    values = []
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column_name not in (reader.fieldnames or []):
            raise ValueError(f"{file_path} 中不存在列: {column_name}")
        for row in reader:
            value = row.get(column_name, "")
            if value is None:
                continue
            value = str(value).strip()
            if value:
                values.append(value)
    return values

def build_ndc_map(file_path: str) -> dict:
    ndc_map = {}
    if not os.path.exists(file_path):
        print(f"⚠️ 未找到 {file_path}，将统一使用 Unknown-NDC")
        return ndc_map

    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "ATC" not in fieldnames or "NDC" not in fieldnames:
            raise ValueError(f"{file_path} 必须包含列 'ATC' 和 'NDC'")

        for row in reader:
            atc = str(row.get("ATC", "")).strip()
            ndc = str(row.get("NDC", "")).strip()
            if not atc or not ndc:
                continue
            if atc not in ndc_map:
                ndc_map[atc] = ndc
    return ndc_map

def unique_preserve_order(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# =========================================================
# Gemini 调用
# =========================================================
def call_gemini(prompt: str, temperature: float = 0.2, max_output_tokens: int = 800) -> str:
    check_api_key()

    params = {"key": GEMINI_API_KEY}
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": 0.8,
            "topK": 10
        }
    }

    resp = requests.post(
        API_URL,
        headers=HEADERS,
        params=params,
        json=payload,
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned. Full response: {data}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError(f"No content parts returned. Full response: {data}")

    texts = []
    for p in parts:
        if "text" in p:
            texts.append(p["text"])

    result = "".join(texts).strip()
    if not result:
        raise ValueError(f"Empty text returned. Full response: {data}")

    return result

def call_with_retry(prompt: str, code: str, fail_log_file: str,
                    temperature: float = 0.2, max_output_tokens: int = 800) -> str:
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return call_gemini(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        except Exception as e:
            print(f"⚠️ Error for {code} (attempt {attempt}/{MAX_RETRY}): {e}")
            if attempt < MAX_RETRY:
                time.sleep(RETRY_DELAY)

    append_fail_code(fail_log_file, code)
    return f"ERROR: Failed after {MAX_RETRY} attempts"

# =========================================================
# 1) Drug 描述生成
# =========================================================
def process_drug_descriptions():
    print("\n🚀 Processing DRUG codes...")

    if not os.path.exists(DRUG_INPUT_CSV):
        print(f"❌ Input file not found: {DRUG_INPUT_CSV}")
        return

    try:
        atc_codes_all = read_csv_column(DRUG_INPUT_CSV, "code")
    except Exception as e:
        print(f"❌ 读取药物输入文件失败: {e}")
        return

    try:
        ndc_map = build_ndc_map(NDC_ATC_MAP_CSV)
    except Exception as e:
        print(f"❌ 构建 NDC 映射失败: {e}")
        return

    atc_codes_all = unique_preserve_order(atc_codes_all)
    completed = read_completed_codes(DRUG_OUTPUT_CSV, "ATC")
    pending = [c for c in atc_codes_all if c not in completed]

    print(f"Total: {len(atc_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")

    ensure_csv_header(DRUG_OUTPUT_CSV, ["ATC", "NDC", "DESCRIPTION"])

    with open(DRUG_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for atc_code in tqdm(pending, desc="Generating drug descriptions"):
            ndc_code = ndc_map.get(atc_code, "Unknown-NDC")
            prompt = DRUG_PROMPT_TMPL.format(atc3_code=atc_code, ndc_code=ndc_code)
            desc = call_with_retry(
                prompt=prompt,
                code=atc_code,
                fail_log_file=DRUG_FAIL_LOG,
                temperature=0.1,
                max_output_tokens=800
            )
            writer.writerow([atc_code, ndc_code, clean_text(desc)])
            f.flush()
            time.sleep(REQUEST_DELAY)

    print(f"✅ Drug descriptions saved to '{DRUG_OUTPUT_CSV}'")

# =========================================================
# 2) Diagnosis 描述生成
# =========================================================
def process_diagnosis_descriptions():
    print("\n🚀 Processing DIAGNOSIS codes...")

    if not os.path.exists(VOCAB_PKL):
        print(f"❌ File not found: {VOCAB_PKL}")
        return

    try:
        with open(VOCAB_PKL, "rb") as f:
            vocabs = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取 official_vocabs.pkl 失败: {e}")
        return

    if "diagnoses" not in vocabs:
        print("❌ official_vocabs.pkl must contain key 'diagnoses'")
        return

    diag_codes_all = []
    for x in vocabs["diagnoses"]:
        if x is None:
            continue
        code = str(x).strip()
        if code and code.lower() != "nan":
            diag_codes_all.append(code)

    diag_codes_all = sorted(set(diag_codes_all))
    completed = read_completed_codes(DIAG_OUTPUT_CSV, "ICD9_CODE")
    pending = [c for c in diag_codes_all if c not in completed]

    print(f"Total: {len(diag_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")

    ensure_csv_header(DIAG_OUTPUT_CSV, ["ICD9_CODE", "DESCRIPTION"])

    with open(DIAG_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for icd9_code in tqdm(pending, desc="Generating diagnosis descriptions"):
            prompt = DIAGNOSIS_PROMPT_TMPL.format(icd9_code=icd9_code)
            desc = call_with_retry(
                prompt=prompt,
                code=icd9_code,
                fail_log_file=DIAG_FAIL_LOG,
                temperature=0.2,
                max_output_tokens=900
            )
            writer.writerow([icd9_code, clean_text(desc)])
            f.flush()
            time.sleep(REQUEST_DELAY)

    print(f"✅ Diagnosis descriptions saved to '{DIAG_OUTPUT_CSV}'")

# =========================================================
# 3) Procedure 描述生成
# =========================================================
def process_procedure_descriptions():
    print("\n🚀 Processing PROCEDURE codes...")

    if not os.path.exists(PROC_INPUT_CSV):
        print(f"❌ Input file not found: {PROC_INPUT_CSV}")
        return

    try:
        proc_codes_all = read_csv_column(PROC_INPUT_CSV, "ICD9_CODE")
    except Exception as e:
        print(f"❌ 读取 procedure 输入文件失败: {e}")
        return

    proc_codes_all = sorted(set(proc_codes_all))
    completed = read_completed_codes(PROC_OUTPUT_CSV, "ICD9_CODE")
    pending = [c for c in proc_codes_all if c not in completed]

    print(f"Total: {len(proc_codes_all)}, Completed: {len(completed)}, Pending: {len(pending)}")

    ensure_csv_header(PROC_OUTPUT_CSV, ["ICD9_CODE", "DESCRIPTION"])

    with open(PROC_OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for icd9_code in tqdm(pending, desc="Generating procedure descriptions"):
            prompt = PROCEDURE_PROMPT_TMPL.format(icd9_code=icd9_code)
            desc = call_with_retry(
                prompt=prompt,
                code=icd9_code,
                fail_log_file=PROC_FAIL_LOG,
                temperature=0.2,
                max_output_tokens=900
            )
            writer.writerow([icd9_code, clean_text(desc)])
            f.flush()
            time.sleep(REQUEST_DELAY)

    print(f"✅ Procedure descriptions saved to '{PROC_OUTPUT_CSV}'")

# =========================================================
# 主函数
# =========================================================
def main():
    check_api_key()

    print("====================================================")
    print("Starting generation pipeline with Gemini 2.5 Flash-Lite")
    print("====================================================")

    process_drug_descriptions()
    process_diagnosis_descriptions()
    process_procedure_descriptions()

    print("\n🎉 All tasks completed!")

if __name__ == "__main__":
    main()