import pandas as pd
import pickle

diag_file = "MIMIC3_DIAGNOSES_Embeddings_512_ID.csv"
proc_file = "MIMIC3_PROCEDURES_Embeddings_512_ID.csv"

# 读取 ICD9_CODE 列
diag_codes = pd.read_csv(diag_file, dtype=str)["ICD9_CODE"].dropna().unique().tolist()
proc_codes = pd.read_csv(proc_file, dtype=str)["ICD9_CODE"].dropna().unique().tolist()

official_vocabs = {
    "diagnoses": sorted(diag_codes),
    "procedures": sorted(proc_codes)
}

with open("official_vocabs.pkl", "wb") as f:
    pickle.dump(official_vocabs, f)

print("🎉 official_vocabs.pkl 已成功重建！")
print(f"诊断编码数量: {len(diag_codes)}")
print(f"操作编码数量: {len(proc_codes)}")
