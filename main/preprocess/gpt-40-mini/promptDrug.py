import os
import time
import csv
import requests
import pandas as pd
from tqdm import tqdm

# =========================
# 1. Configuration
# =========================
INPUT_CSV       = "mimic3_drugs_mapping.csv"       # your ATC list with ATC column
OUTPUT_CSV      = "MIMIC3_Drug_Descriptions_ID_HCLFinal.csv" # filled descriptions
NDC_ATC_MAP_CSV = "ndc_atc_map_level3.csv"     # must contain ATC and NDC columns

API_URL         = "https://api.zhizengzeng.com/v1/chat/completions"
OPENAI_API_KEY  = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # your key here
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}
REQUEST_DELAY   = 2  # seconds

# DRUG_PROMPT_TMPL = (
#     "You are a distinguished clinical pharmacologist and expert in cheminformatics. Your objective is to generate a comprehensive, machine-readable description for a drug class, identified by its ATC-3 code. This description must bridge its clinical application with its underlying pharmacological properties, enabling an AI model to learn complex drug-drug,drug-procedures and drug-disease relationships."
#     "For the medication with ATC level-3 code: \"{atc3_code}\" (representative NDC: \"{ndc_code}\")\n\n"
#     "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow.Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
#     "1.State Anatomical Main Group (ATC Level 1): State the drug's broader anatomical main group (e.g., C for Cardiovascular system).\n"
#     "2.Define Therapeutic Context (ATC Level 2): Describe the therapeutic subgroup (e.g., C07 for Beta blocking agents). List the typical diseases it treats (with ICD-9-CM codes) and associated clinical procedures (with ICD-9-CM procedure codes).\n"
#     "3.Detail Pharmacological Profile and Interactions (ATC Levels 3-5): Detail the specific pharmacological class, its precise mechanism of action (MoA), common synergistic co-prescriptions (with ATC-3 codes), and critical drug-drug interactions (DDIs).\n"
#     "For example, your output should look like this: 'C07AB selective beta-blocking agents (ATC-3: C07AB) belong to the cardiovascular system (ATC-1: C) and are primarily indicated for hypertension (ICD-9-CM: 401.9) and angina (ICD-9-CM: 413.9). They reduce sympathetic drive by blocking cardiac beta-1 receptors…… Commonly co-prescribed with ACE inhibitors (ATC-3: C09AA)…… Concurrent use with calcium channel blockers (ATC-3: C08DA) should be avoided due to bradycardia risk……"
# )

DRUG_PROMPT_TMPL = (
    "You are a distinguished clinical pharmacologist and expert in cheminformatics. Your objective is to generate a comprehensive, machine-readable description for a drug class, identified by its ATC-3 code. "
    "This description must bridge its clinical application with its underlying pharmacological properties, enabling an AI model to learn complex drug-drug,drug-procedures and drug-disease relationships."
    "For the medication with ATC level-3 code: \"{atc3_code}\" (representative NDC: \"{ndc_code}\")\n\n"
    "Synthesize the following information into a single continuous paragraph of formal medical text.Never use quotation marks for any terms, names, or codes. "
    "Do not use bullet points, headings, or numbered lists. If a specific point (e.g., common procedures) is not clinically relevant or applicable, omit it from the description to maintain a natural flow."
    "Embed all medical codes naturally within the text immediately after the relevant term, using the formats 'ICD-9-CM: [code]' and 'ATC-3: [code]'.\n"
    "1.State Anatomical Main Group (ATC Level 1): State the drug's broader anatomical main group (e.g., C for Cardiovascular system).\n" 
    "2.Define Therapeutic Context (ATC Level 2): Describe the therapeutic subgroup (e.g., C07 for Beta blocking agents). "
    "List the typical diseases it treats (with ICD-9-CM codes) and associated clinical procedures (with ICD-9-CM procedure codes).\n"
    "3.Detail Pharmacological Profile and Interactions (ATC Levels 3): Detail the specific pharmacological class, its precise mechanism of action (MoA), "
    "common synergistic co-prescriptions (with ATC-3 codes), and critical drug-drug interactions (DDIs).\n"
    "For example, your output should look like this: "
    "'C07AB selective beta-blocking agents (ATC-3: C07AB) belong to the cardiovascular system (ATC-1: C) and are primarily indicated for hypertension (ICD-9-CM: 401.9) and angina (ICD-9-CM: 413.9). They reduce sympathetic drive by blocking cardiac beta-1 receptors…… Commonly co-prescribed with ACE inhibitors (ATC-3: C09AA)…… Concurrent use with calcium channel blockers (ATC-3: C08DA) should be avoided due to bradycardia risk……"
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