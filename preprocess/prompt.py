import pandas as pd
import pickle
import os
import csv
import time
import requests  # 使用标准的同步请求库
from tqdm import tqdm  # 使用标准tqdm

# ==============================================================================
# 1. 配置区域 (Configuration)
# ==============================================================================
# 包含官方编码的缓存文件路径
VOCAB_CACHE_PATH = "official_vocabs.pkl"

# API 设置
API_URL = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # 您的 API Key
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}

# 性能设置
# 在同步模式下，这个参数不再是并发数，而是两次请求间的固定延迟（秒）
REQUEST_DELAY = 2

# ==============================================================================
# 2. Prompt 模板 (保持不变)
# ==============================================================================
DIAGNOSIS_PROMPT_TEMPLATE = (
    "You are an expert clinical informatician. Your task is to provide a detailed and clinically relevant description for a given ICD-9 procedure code. "
    "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
    "For the ICD-9 procedure code: \"{icd9_code}\"\n\n"
    "Please provide a description that covers the following aspects in a concise paragraph:\n"
    "1. Procedure Definition: Briefly describe what the procedure is.\n"
    "2. Primary Indications: For what primary diagnoses or conditions is this procedure typically performed? This directly links the procedure back to the patient's problems.\n"
    "3. Purpose of Procedure: What is the main goal of this procedure (e.g., diagnostic, therapeutic, palliative)?\n"
    "4. Common Post-Procedure Medications: What types of medications are commonly prescribed after this procedure? Consider medications for pain management, infection prevention, management of the underlying condition, or preventing complications (e.g., 'analgesics', 'prophylactic antibiotics', 'thromboprophylaxis').\n\n"
    "Please respond in a single continuous paragraph without any line breaks or additional commentary."
)

PROCEDURE_PROMPT_TEMPLATE = (
    "You are an expert clinical informatician. Your task is to provide a detailed and clinically relevant description for a given ICD-9 procedure code. "
    "The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
    "For the ICD-9 procedure code: \"{icd9_code}\"\n\n"
    "Please provide a description that covers the following aspects in a concise paragraph:\n"
    "1. Procedure Definition & Purpose: Briefly describe the procedure and its main goal (e.g., diagnostic, therapeutic).\n"
    "2. Primary Indications (Diagnoses): For what primary ICD-9 diagnoses is this procedure typically performed?\n"
    "3. Common Post-Procedure Medications: What classes of medications(with drug NDC/ATC-3 codes) are commonly prescribed immediately following this procedure for recovery and management?\n"
    "4. Key Contraindications or Cautions: Are there specific drugs or drug classes(with drug NDC/ATC-3 codes) that are often contraindicated or must be stopped prior to or immediately after this procedure?\n\n"
    "Use one continuous paragraph using formal medical language. Do not use bullet points or headings. Embed the actual codes (ATC-3, ICD-9-CM, ICD-9, NDC) naturally in the text as appropriate."
)


# ==============================================================================
# 3. 同步 API 调用逻辑
# ==============================================================================
def get_single_description_sync(icd9_code, prompt_template):
    """同步发送单个请求，并处理API返回的逻辑错误。"""
    prompt = prompt_template.format(icd9_code=icd9_code)
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data, timeout=60)
        response.raise_for_status()  # 检查HTTP错误
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            description = result["choices"][0]["message"]["content"]
            return description
        else:
            error_message = result.get("error", {}).get("message", "Unknown API Error")
            print(f"\n  - API Logic Error for code {icd9_code}: {error_message}")
            return f"ERROR: {error_message}"

    except requests.exceptions.RequestException as e:
        print(f"\n  - Network Error for code {icd9_code}: {e}")
        return "ERROR: Network issue"


# ==============================================================================
# 4. 主处理工作流 (同步版)
# ==============================================================================
def generate_all_descriptions_sync():
    """主同步工作流，用于生成所有描述。"""

    if not os.path.exists(VOCAB_CACHE_PATH):
        print(f"❌ Error: Vocabulary cache file '{VOCAB_CACHE_PATH}' not found.")
        print("Please run the 'generate_official_vocabs.py' script first.")
        return

    with open(VOCAB_CACHE_PATH, 'rb') as f:
        official_vocabs = pickle.load(f)

    for code_type, prompt_template in [("diagnoses", DIAGNOSIS_PROMPT_TEMPLATE),
                                       ("procedures", PROCEDURE_PROMPT_TEMPLATE)]:

        output_csv = f"{code_type.upper()}_Descriptions_Official.csv"
        codes_to_process = official_vocabs[code_type]

        print(f"\n--- 🚀 Starting to generate descriptions for {len(codes_to_process)} {code_type} (SYNC MODE) ---")

        completed_codes = set()
        if os.path.exists(output_csv):
            try:
                df_existing = pd.read_csv(output_csv, dtype={'ICD9_CODE': str})
                if 'ICD9_CODE' in df_existing.columns:
                    # 排除掉之前出错的行，以便重试
                    successful_rows = df_existing[~df_existing['DESCRIPTION'].str.startswith('ERROR', na=False)]
                    completed_codes = set(successful_rows['ICD9_CODE'])
                print(f"👍 Resuming: Found {len(completed_codes)} successfully completed codes in '{output_csv}'.")
            except Exception as e:
                print(f"⚠️ Could not read existing output file '{output_csv}'. Starting from scratch. Error: {e}")

        pending_codes = sorted([code for code in codes_to_process if code not in completed_codes])

        if not pending_codes:
            print(f"✅ All {code_type} descriptions are already generated.")
            continue

        print(
            f"⏳ Pending {len(pending_codes)} codes for {code_type}. Estimated time: {len(pending_codes) * REQUEST_DELAY / 60:.1f} minutes.")

        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not completed_codes:
                writer.writerow(["ICD9_CODE", "DESCRIPTION"])

            # 使用标准的tqdm进度条
            for code in tqdm(pending_codes):
                description = get_single_description_sync(code, prompt_template)
                writer.writerow([code, description])
                f.flush()  # 立即写入磁盘
                time.sleep(REQUEST_DELAY)  # 强制延迟，尊重API速率限制

        print(f"🎉 Finished generating descriptions for {code_type}. Results saved to '{output_csv}'.")


if __name__ == "__main__":
    generate_all_descriptions_sync()