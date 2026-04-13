import os
import time
import csv
import requests
import pandas as pd
from tqdm import tqdm

# =========================
# 1. Configuration
# =========================
INPUT_CSV       = "ATC3_Unique_List.csv"       # your ATC list with ATC column
OUTPUT_CSV      = "ATC3_Drug_Descriptions.csv" # filled descriptions
NDC_ATC_MAP_CSV = "ndc_atc_map_level3.csv"     # must contain ATC and NDC columns

API_URL         = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY  = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # your key here
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}
REQUEST_DELAY   = 2  # seconds
# DRUG_PROMPT_TMPL = (
#     "You are a clinical pharmacist. Your task is to provide a concise summary of various names "
#     "for a given medication, identified by its ATC-3 code and a known generic name.\n\n"
#     "For the medication with ATC-3 code \"{atc}\" (generic name: \"{name}\"):\n"
#     "List its generic name(s), common brand names, and any other relevant forms, "
#     "in one continuous paragraph."
# )
DRUG_PROMPT_TMPL = (
    "You are a senior clinical pharmacologist and biomedical informatics expert. Your task is to generate a clinically rich, machine-readable description for a drug class based on its ATC level-3 code."
    "The description should help a machine learning model understand its usage, indications, and clinical connections."
    "For the medication with ATC level-3 code \"{atc3_code}\", please write a single, continuous paragraph in formal medical language. Your description should logically cover the following aspects in order:\n"
    "1. The pharmacological class and mechanism of action.\n"
    "2. Typical diagnoses it is used to treat (with their ICD-9 codes).\n"
    "3. Common clinical procedures (with their ICD-9 procedure codes) associated with its use.\n"
    "4. Frequent co-prescribed drug classes (with their ATC-3 codes).\n"
    "5. Significant contraindications and patient populations requiring caution.\n"
    "6. Common and clinically relevant adverse effects.\n\n"
    "Do not use bullet points or headings. If a specific point (e.g., associated procedures) is not applicable, omit it from the description. Embed all codes naturally within the text.\n\n"
    "For example, your output should look like this: 'ACE inhibitors (ATC-3: C09AA) act by blocking the conversion of angiotensin I to angiotensin II, leading to vasodilation and reduced aldosterone secretion. They are primarily used for treating hypertension (ICD-9: 401.9) and congestive heart failure (ICD-9: 428.0). These drugs are frequently co-prescribed with diuretics (ATC-3: C03) to enhance blood pressure control. ACE inhibitors are contraindicated in pregnancy and in patients with a history of angioedema. The most common adverse effect is a dry, persistent cough.'"
)



def get_drug_description(atc_code: str, ndc_code: str) -> str:
    prompt = DRUG_PROMPT_TMPL.format(atc3_code=atc_code, ndc_code=ndc_code)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Error for {atc_code}: {e}")
        return f"ERROR: {e}"


def main():
    # 2. Load ATC input list
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        return
    df_atc = pd.read_csv(INPUT_CSV, dtype=str)
    if "ATC" not in df_atc.columns:
        print("❌ Input CSV must have column 'ATC'")
        return

    # 3. Load NDC and generic name mapping
    if not os.path.exists(NDC_ATC_MAP_CSV):
        print(f"❌ Map file not found: {NDC_ATC_MAP_CSV}")
        return
    df_map = pd.read_csv(NDC_ATC_MAP_CSV, dtype=str)

    # 构建 ATC → NDC 映射（仅取第一个 NDC）
    ndc_map = df_map.dropna(subset=["NDC"]) \
                    .drop_duplicates(subset=["ATC"]) \
                    .set_index("ATC")["NDC"].to_dict()

    # 4. Resume support: skip ATC codes already processed
    completed = set()
    if os.path.exists(OUTPUT_CSV):
        df_exist = pd.read_csv(OUTPUT_CSV, dtype=str)
        if "ATC" in df_exist.columns:
            completed = set(df_exist["ATC"])
        print(f"Resuming: {len(completed)} codes already processed.")

    # 5. Open output CSV for append
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        # write header if new
        if not completed:
            writer.writerow(["ATC", "NDC", "DESCRIPTION"])

        # iterate through all ATC codes
        for atc in tqdm(df_atc["ATC"].tolist(), desc="Generating descriptions"):
            if atc in completed:
                continue

            ndc_code = ndc_map.get(atc, "Unknown-NDC")
            desc = get_drug_description(atc, ndc_code)
            writer.writerow([atc, ndc_code, desc])
            fout.flush()
            time.sleep(REQUEST_DELAY)

    print(f"\n✅ All done. Descriptions saved to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()