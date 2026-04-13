import pandas as pd
import requests
import time
import csv
import os
from tqdm import tqdm

# --- ⚙️ 1. Configuration ---
# You can easily change these parameters
INPUT_FILE = "PROCEDURES_Despction_Med.csv"
OUTPUT_FILE = "PROCEDURES_Med_Embedding_512.csv"
MODEL_NAME = "text-embedding-3-small"  # Or "text-embedding-3-large" for higher quality
EMBEDDING_DIM = 128  # Recommended to match LAMRec's internal dimension
BATCH_SIZE = 32  # Process 32 descriptions per API call
API_URL = "https://api.zhizengzeng.com/v1/embeddings"
OPENAI_API_KEY = "sk-zk249ad191da5206f1ca5d3e7e6fc9bc527683091ad9c28b"  # <-- Replace with your key

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + OPENAI_API_KEY
}


# --- 🧠 2. Enhanced Embedding Function (Handles Batches) ---
def get_embeddings_batch(texts: list[str]):
    """
    Sends a batch of texts to the embedding API.
    """
    # The API expects a list of strings as input
    data = {
        "model": MODEL_NAME,
        "input": texts,
        "dimensions": EMBEDDING_DIM
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    response.raise_for_status()  # Will raise an exception for HTTP errors

    # The response contains a list of embedding data, one for each input text
    embeddings = [item["embedding"] for item in response.json()["data"]]
    return embeddings


# --- 🚀 3. Main Processing Logic with Batching ---
def generate_embeddings_in_batches():
    """
    Reads the description file, generates embeddings in batches for performance, and saves them.
    """
    # Load the entire input file
    try:
        df_input = pd.read_csv(INPUT_FILE, dtype={"ICD9_CODE": str})
        df_input.rename(columns={'Despction': 'DESCRIPTION', 'DESCRIPTION': 'DESCRIPTION'},
                        inplace=True)  # Standardize column name
        df_input.dropna(subset=['DESCRIPTION'], inplace=True)
    except FileNotFoundError:
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # Check for existing output file to resume processing
    processed_codes = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE)
            processed_codes = set(df_existing["ICD9_CODE"].astype(str))
            print(f"👍 Resuming: {len(processed_codes)} codes already processed.")
        except Exception as e:
            print(f"⚠️ Could not read existing output file, starting from scratch. Error: {e}")

    # Filter out already processed descriptions
    df_todo = df_input[~df_input['ICD9_CODE'].isin(processed_codes)].copy()

    if df_todo.empty:
        print("✅ All descriptions have already been processed.")
        return

    print(f"⏳ Processing {len(df_todo)} new descriptions in batches of {BATCH_SIZE}...")

    # Open the output file in append mode
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header only if the file is new
        if not processed_codes:
            writer.writerow(["ICD9_CODE", "Embedding"])

        # Process the dataframe in chunks (batches)
        for i in tqdm(range(0, len(df_todo), BATCH_SIZE)):
            batch_df = df_todo.iloc[i:i + BATCH_SIZE]

            codes_batch = batch_df['ICD9_CODE'].tolist()
            texts_batch = batch_df['DESCRIPTION'].tolist()

            # Pre-process texts: handle "ERROR" and clean up
            cleaned_texts = []
            error_indices = set()
            for idx, text in enumerate(texts_batch):
                if isinstance(text, str) and text.strip().upper() == "ERROR":
                    error_indices.add(idx)
                    # Add a placeholder; it won't be sent to the API
                    cleaned_texts.append("no description available")
                else:
                    # Basic cleaning: strip whitespace
                    cleaned_texts.append(str(text).strip())

            # Get embeddings for the valid texts in the batch
            try:
                if len(cleaned_texts) > len(error_indices):
                    embeddings_batch = get_embeddings_batch(cleaned_texts)
                else:  # Handle case where the whole batch is errors
                    embeddings_batch = []
            except Exception as e:
                print(f"\n❌ API Error on batch starting at index {i}: {e}. Skipping batch.")
                # Write error rows for the entire batch
                for code in codes_batch:
                    writer.writerow([code, [0.0] * EMBEDDING_DIM])
                continue

            # Merge results and write to CSV
            embedding_idx = 0
            for i, code in enumerate(codes_batch):
                if i in error_indices:
                    writer.writerow([code, [0.0] * EMBEDDING_DIM])
                else:
                    writer.writerow([code, embeddings_batch[embedding_idx]])
                    embedding_idx += 1

            f.flush()  # Ensure the batch is written to disk

    print(f"🎉 Embedding generation complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_embeddings_in_batches()