import pandas as pd
import requests
import time
import csv
import os
from tqdm import tqdm
from typing import List  # 导入 List

# --- ⚙️ 1. Configuration ---
INPUT_FILE = "MIMIC3_Drug_Descriptions_ID.csv"              # 输入文件：ATC + DESCRIPTION
OUTPUT_FILE = "MIMIC3_Drug_Embeddings_512_ID.csv"     # 输出文件
MODEL_NAME = "text-embedding-3-large"
EMBEDDING_DIM = 512
BATCH_SIZE = 32
API_URL = "https://api.zhizengzeng.com/v1/embeddings"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}

# --- 🧠 2. Embedding API call ---
def get_embeddings_batch(texts: List[str]):  # 使用 List[str]
    data = {
        "model": MODEL_NAME,
        "input": texts,
        "dimensions": EMBEDDING_DIM
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()
    embeddings = [item["embedding"] for item in response.json()["data"]]
    return embeddings

# --- 🚀 3. Main ---
def generate_embeddings_in_batches():
    try:
        df_input = pd.read_csv(INPUT_FILE, dtype=str)
        df_input.dropna(subset=["DESCRIPTION"], inplace=True)
    except FileNotFoundError:
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # 支持断点续写
    processed_codes = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            processed_codes = set(df_existing["ATC"].astype(str))
            print(f"👍 Resuming: {len(processed_codes)} codes already processed.")
        except Exception as e:
            print(f"⚠️ Failed to read output file. Starting fresh. Error: {e}")

    df_todo = df_input[~df_input["ATC"].isin(processed_codes)].copy()
    if df_todo.empty:
        print("✅ All descriptions have already been processed.")
        return

    print(f"⏳ Processing {len(df_todo)} ATC descriptions in batches of {BATCH_SIZE}...")

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not processed_codes:
            writer.writerow(["ATC", "Embedding"])

        for i in tqdm(range(0, len(df_todo), BATCH_SIZE)):
            batch_df = df_todo.iloc[i:i + BATCH_SIZE]
            atc_batch = batch_df["ATC"].tolist()
            desc_batch = batch_df["DESCRIPTION"].tolist()

            cleaned_texts = []
            error_indices = set()
            for idx, text in enumerate(desc_batch):
                if isinstance(text, str) and text.strip().upper().startswith("ERROR"):
                    error_indices.add(idx)
                    cleaned_texts.append("no description available")
                else:
                    cleaned_texts.append(text.strip())

            try:
                if len(cleaned_texts) > len(error_indices):
                    embeddings_batch = get_embeddings_batch(cleaned_texts)
                else:
                    embeddings_batch = []
            except Exception as e:
                print(f"\n❌ API Error at batch {i}: {e}")
                for atc in atc_batch:
                    writer.writerow([atc, [0.0] * EMBEDDING_DIM])
                continue

            # 写入结果
            embedding_idx = 0
            for i, atc in enumerate(atc_batch):
                if i in error_indices:
                    writer.writerow([atc, [0.0] * EMBEDDING_DIM])
                else:
                    writer.writerow([atc, embeddings_batch[embedding_idx]])
                    embedding_idx += 1

            f.flush()

    print(f"🎉 Embedding generation complete. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_embeddings_in_batches()
