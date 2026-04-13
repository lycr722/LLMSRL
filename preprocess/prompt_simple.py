import pandas as pd
import requests
import time
import csv
import os

# API 设置 (请替换为您的KEY)
API_URL = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # <-- 请在这里填入您的 API Key

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}


# ==============================================================================
# 1. 针对“诊断 (Diagnosis)” 的 PROMPT (已修改为朴素版本)
# ==============================================================================
def get_diagnosis_description(icd9_code):
    """
    使用朴素提示 (Naive Prompt) 调用大模型获取诊断描述。
    这是为消融实验设计的基线版本。
    """
    # --- 您原来的复杂提示词已被注释掉，以便随时恢复 ---
    """
    prompt = (
        "You are a senior clinician-scientist and biomedical informatician. Your task is to provide a detailed clinically relevant description for a given ICD-9-CM diagnosis code. The goal is to generate a description that helps a machine learning model understand the impact of the diagnosis on the recommended drug combination."
        "For ICD-9-CM diagnosis codes\"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Core Definition: Briefly define the diagnosis and its typical clinical impact."
        "2. Common co-existing diseases (with their ICD-9-CM diagnosis codes) that influence polypharmacy. "
        "3. Major Contraindications: Specify drugs or drug classes (with ATC-3 codes) that should be avoided or used with extreme caution.Crucially, explain why "
        "4. Relevant medical or surgical procedures (with ICD-9-CM procedure codes) commonly associated with this diagnosis. "
        "5. Treatment goals and typical drug classes: What are the primary treatment goals (e.g., symptom management, slowing progression, cure)? Based on these goals, what are the first-line and second-line drug classes (with ATC-3 codes) typically prescribed, and what is the reasoning?"
        "For example, your output should look like this: 'Type 2 diabetes mellitus (ICD-9-CM: 250.00) is a chronic metabolic disorder...'"
    )
    """

    # --- 新的朴素提示词 (Naive Prompt) ---
    prompt = (
        "Provide a general description for the medical diagnosis "
        "associated with the ICD-9 code: '{icd9_code}'."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,
        "stream": False
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==============================================================================
# 2. 针对“治疗过程 (Procedure)” 的 PROMPT (已修改为朴素版本)
# ==============================================================================
def get_procedure_description(icd9_code):
    """
    使用朴素提示 (Naive Prompt) 调用大模型获取手术描述。
    这是为消融实验设计的基线版本。
    """
    # --- 您原来的复杂提示词已被注释掉，以便随时恢复 ---
    """
    prompt=(
        "You are an expert clinical informatician and surgical pharmacologist. Your task is to provide a detailed and clinically relevant description for a given ICD-9-CM procedure code.The goal is to generate a description that helps a machine learning model understand the context and purpose of this procedure, especially its relationship to subsequent medication needs. "
        "For the ICD-9-CM procedure code: \"{icd9_code}\"\n\n"
        "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
        "1. Procedure Definition & Purpose: Briefly describe the procedure and its main goal (e.g., diagnostic, therapeutic).\n"
        "2. Primary Indications (Diagnoses): For what primary ICD-9-CM diagnoses is this procedure typically performed?\n"
        "3. Pre-procedure: What drug classes (with ATC-3 codes) often need to be held or adjusted before the procedure and why (e.g., 'discontinuation of anticoagulants (ATC-3: B01AA) to mitigate bleeding risk')?"
        "4. Post-procedure: What are the common short-term medications (with ATC-3 codes) prescribed immediately after, and for what purpose (e.g., 'for prophylaxis, pain management, or recovery support')?"
        "5. Key Contraindications or Cautions: Are there specific drugs or drug classes(with drug ATC-3 codes) that are often contraindicated or must be stopped prior to or immediately after this procedure?\n\n"
        "For example, your output should look like this: 'This procedure is primarily indicated for symptomatic cholelithiasis (ICD-9-CM: 574.20)...'"
    )
    """

    # --- 新的朴素提示词 (Naive Prompt) ---
    prompt = (
        "Provide a general description for the medical procedure "
        "associated with the ICD-9 code: '{icd9_code}'."
    )

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt.format(icd9_code=icd9_code)}],
        "temperature": 0.5,
        "stream": False
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ==============================================================================
# 3. 主逻辑 (无需修改)
# ==============================================================================
def process_codes(code_type='diagnosis'):
    """
    主处理函数。
    :param code_type: 'diagnosis' 或 'procedure'
    """
    if code_type == 'diagnosis':
        input_csv = "DIAGNOSES_ICD.csv"
        output_csv = "DIAGNOSES_Description_Simple.csv"  # 建议使用新文件名以避免覆盖
        api_function = get_diagnosis_description
        print("🚀 Starting to process DIAGNOSIS codes with NAIVE prompts...")
    elif code_type == 'procedure':
        input_csv = "PROCEDURES_ICD.csv"
        # input_csv = "mimic3_procedures_mapping.csv"
        output_csv = "PROCEDURES_Description_Simple.csv"  # 建议使用新文件名以避免覆盖
        api_function = get_procedure_description
        print("🚀 Starting to process PROCEDURE codes with NAIVE prompts...")
    else:
        print("❌ Invalid code_type. Please choose 'diagnosis' or 'procedure'.")
        return

    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"❌ Input file not found: {input_csv}")
        return

    # 读取原始 ICD9 代码
    df = pd.read_csv(input_csv, dtype={"ICD9_CODE": str})
    icd9_codes_all = sorted(set(df["ICD9_CODE"].dropna()))

    # 读取已完成的代码（如果有）
    completed_codes = set()
    # 第一次运行时创建文件头
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ICD9_CODE", "DESCRIPTION"])
    else:
        try:
            existing_df = pd.read_csv(output_csv, dtype={"ICD9_CODE": str})
            if "ICD9_CODE" in existing_df.columns:
                completed_codes = set(existing_df["ICD9_CODE"].dropna())
        except Exception as e:
            print(f"⚠️ Warning: Could not read existing output file. Starting from scratch. Error: {e}")

    # 过滤出未完成的代码
    pending_codes = [code for code in icd9_codes_all if code not in completed_codes]
    print(
        f"Total unique codes: {len(icd9_codes_all)}, Completed: {len(completed_codes)}, Pending: {len(pending_codes)}")

    if not pending_codes:
        print("✅ All codes have been processed. Exiting.")
        return

    # 以 append 模式继续写入
    with open(output_csv, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx, code in enumerate(pending_codes):
            # 防止空代码导致错误
            if not code or pd.isna(code):
                continue

            print(f"Processing [{idx + 1}/{len(pending_codes)}]: {code}")
            try:
                description = api_function(code)
                print(f"  -> Got description: {description[:80]}...")
                writer.writerow([code, description])
            except Exception as e:
                print(f"  ❌ Error on code {code}: {e}")
                writer.writerow([code, "ERROR"])

            # 尊重API的速率限制
            time.sleep(2)

    print(f"✅ Processing finished for {code_type}.")


if __name__ == "__main__":
    # 您想处理哪种类型的代码？ 'diagnosis' 或 'procedure'
    # =======================================================
    # target_code_type = 'diagnosis'
    target_code_type = 'procedure'
    # =======================================================

    process_codes(target_code_type)