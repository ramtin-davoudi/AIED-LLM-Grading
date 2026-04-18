
import json
import re
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# =========================
# Configuration
# =========================
OPENAI_API_KEY = "API_Key"

DATASET_NAME = "nkazi/SciEntsBank"
SPLIT = "test_ua"
MODEL = "gpt-5.2"

MAX_ROWS = 540     
POLL_SECONDS = 15

OUT_DIR = Path("scientsbank_batch_run")
OUT_DIR.mkdir(exist_ok=True)

BATCH_INPUT_PATH = OUT_DIR / "batch_input.jsonl"
RAW_RESULTS_PATH = OUT_DIR / "batch_output_raw.jsonl"
RAW_ERRORS_PATH = OUT_DIR / "batch_errors_raw.jsonl"
OUTPUT_CSV_PATH = OUT_DIR / f"scientsbank_{SPLIT}_{MODEL}_with_llm_labels_zero_shot.csv"



# =========================
# Prompt
# =========================
SYSTEM_PROMPT = """
You are an educational short-answer grader.

You will receive:
- a science question
- a reference answer
- a student answer

Return exactly one numeric label:

0 = correct
1 = contradictory
2 = partially_correct_incomplete
3 = irrelevant
4 = non_domain

Rules:
Return ONLY the number.
Do not explain.
"""


# =========================
# Helpers
# =========================
def build_prompt(row):

    return f"""
{SYSTEM_PROMPT}

Question:
{row['question']}

Reference answer:
{row['reference_answer']}

Student answer:
{row['student_answer']}
"""


def clean_model_label(text):

    if text is None:
        return None

    text = str(text).strip()

    match = re.search(r"\b([0-4])\b", text)

    if match:
        return int(match.group(1))

    return None


# =========================
# Create Batch Request
# =========================
def make_batch_request(row):

    prompt = build_prompt(row)

    return {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": 16
        }
    }


# =========================
# Step 1 Load dataset
# =========================
print("Loading dataset")

dataset = load_dataset(DATASET_NAME)[SPLIT]

df = dataset.to_pandas()

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)

print("Rows:", len(df))


# =========================
# Step 2 Create JSONL
# =========================
print("Building batch JSONL")

with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:

    for _, row in tqdm(df.iterrows(), total=len(df)):

        req = make_batch_request(row)

        f.write(json.dumps(req) + "\n")


# =========================
# Step 3 Create OpenAI client
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Step 4 Upload batch file
# =========================
print("Uploading batch file")

with open(BATCH_INPUT_PATH, "rb") as f:

    uploaded = client.files.create(
        file=f,
        purpose="batch"
    )

print("File uploaded:", uploaded.id)


# =========================
# Step 5 Create batch job
# =========================
print("Creating batch job")

batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/responses",
    completion_window="24h"
)

print("Batch id:", batch.id)


# =========================
# Step 6 Poll job status
# =========================
terminal_states = {"completed", "failed", "cancelled", "expired"}

while True:

    batch = client.batches.retrieve(batch.id)

    print("Batch status:", batch.status)

    if batch.status in terminal_states:
        break

    time.sleep(POLL_SECONDS)


# =========================
# Step 7 Download results
# =========================
if batch.output_file_id:

    print("Downloading results")

    content = client.files.content(batch.output_file_id)

    RAW_RESULTS_PATH.write_text(content.text)

else:

    print("No output file produced")


# =========================
# Step 8 Download errors
# =========================
if batch.error_file_id:

    print("Downloading error file")

    err = client.files.content(batch.error_file_id)

    RAW_ERRORS_PATH.write_text(err.text)

    print("\nError file content:")

    print(err.text)


# =========================
# Step 9 Parse predictions
# =========================
pred_map = {}

if RAW_RESULTS_PATH.exists():

    with open(RAW_RESULTS_PATH) as f:

        for line in f:

            obj = json.loads(line)

            cid = obj["custom_id"]

            try:

                text = obj["response"]["body"]["output"][0]["content"][0]["text"]

            except Exception:

                text = None

            pred_map[cid] = clean_model_label(text)


# =========================
# Step 10 Merge with dataframe
# =========================
df["llm_label"] = df["id"].astype(str).map(pred_map)

df["llm_matches_gold"] = df["llm_label"] == df["label"]


# =========================
# Step 11 Save results
# =========================
df.to_csv(OUTPUT_CSV_PATH, index=False)

print("Saved results to:", OUTPUT_CSV_PATH)


# =========================
# Step 12 Quick evaluation
# =========================
valid = df["llm_label"].notna().sum()

print("Predictions parsed:", valid)

if valid > 0:

    acc = df.loc[df["llm_label"].notna(), "llm_matches_gold"].mean()

    print("Accuracy:", acc)


# Analysis
y_true = df["label"]
y_pred = df["llm_label"]

mask = y_pred.notna()

y_true = y_true[mask]
y_pred = y_pred[mask]

print("\nAccuracy:", accuracy_score(y_true, y_pred))

p, r, f, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro"
)

print("Precision:", p)
print("Recall:", r)
print("F1:", f)

print("\nClassification report")
print(classification_report(y_true, y_pred))

print("\nConfusion matrix")
print(confusion_matrix(y_true, y_pred))