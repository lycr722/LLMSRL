import pandas as pd
import time
import os
from prompt_PROCEDURE import get_diagnosis_description, get_procedure_description, call_with_retry

def retry_and_replace(csv_file, code_type="diagnosis"):
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file, dtype={"ICD9_CODE": str})
    fail_mask = df["DESCRIPTION"].str.contains("ERROR", na=False)
    failed_codes = df.loc[fail_mask, "ICD9_CODE"].tolist()

    print(f"🔍 Found {len(failed_codes)} failed codes in {csv_file}")

    if code_type == "diagnosis":
        api_func = get_diagnosis_description
        fail_log = "FAILED_DIAG_CODES.txt"
    else:
        api_func = get_procedure_description
        fail_log = "FAILED_PROC_CODES.txt"

    for code in failed_codes:
        print(f"⏳ Retrying {code} ...")
        desc = call_with_retry(api_func, code, fail_log)
        df.loc[df["ICD9_CODE"] == code, "DESCRIPTION"] = desc
        time.sleep(2)

    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"🎉 Retry complete. Updated file saved: {csv_file}")


if __name__ == "__main__":
    # # 重试 diagnosis
    # retry_and_replace("DIAGNOSES_Description_ID.csv", "diagnosis")

    # 重试 procedure
    retry_and_replace("PROCEDURES_Description_ID.csv", "procedure")
