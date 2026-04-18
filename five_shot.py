import json
import re
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# =========================
# Configuration
# =========================
OPENAI_API_KEY = "API_KEY"

DATASET_NAME = "nkazi/SciEntsBank"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test_ua"
MODEL = "gpt-5.2"   # change to gpt-5.2 for the second run

MAX_ROWS = 540    
POLL_SECONDS = 15
RANDOM_SEED = 42

OUT_DIR = Path("scientsbank_batch_run")
OUT_DIR.mkdir(exist_ok=True)

FEW_SHOT_PATH = OUT_DIR / "few_shot_examples.json"
BATCH_INPUT_PATH = OUT_DIR / f"batch_input_{MODEL}_five_shot.jsonl"
RAW_RESULTS_PATH = OUT_DIR / f"batch_output_raw_{MODEL}_five_shot.jsonl"
RAW_ERRORS_PATH = OUT_DIR / f"batch_errors_raw_{MODEL}_five_shot.jsonl"
OUTPUT_CSV_PATH = OUT_DIR / f"scientsbank_{TEST_SPLIT}_{MODEL}_with_llm_labels_five_shot.csv"

LABEL_MAP = {
    0: "correct",
    1: "contradictory",
    2: "partially_correct_incomplete",
    3: "irrelevant",
    4: "non_domain",
}

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
""".strip()


# =========================
# Helpers
# =========================
def clean_model_label(text):
    if text is None:
        return None
    text = str(text).strip()
    match = re.search(r"\b([0-4])\b", text)
    return int(match.group(1)) if match else None


def select_and_save_few_shot_examples(train_df, few_shot_path, random_seed=42):
    """
    Select one fixed example per label from the training set and save them.
    If the file already exists, load and return it instead.
    """
    if few_shot_path.exists():
        with open(few_shot_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        print(f"Loaded existing few-shot examples from: {few_shot_path}")
        return examples

    examples = []
    for label in range(5):
        subset = train_df[train_df["label"] == label].sample(
            n=1, random_state=random_seed
        )
        row = subset.iloc[0]
        examples.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "student_answer": row["student_answer"],
                "label": int(row["label"]),
            }
        )

    with open(few_shot_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved fixed few-shot examples to: {few_shot_path}")
    return examples


def format_few_shot_examples(examples):
    """
    Format the 5 few-shot examples into the prompt.
    """
    blocks = []
    for i, ex in enumerate(examples, start=1):
        block = f"""
Example {i}:
Question:
{ex['question']}

Reference answer:
{ex['reference_answer']}

Student answer:
{ex['student_answer']}

Correct label:
{ex['label']}
""".strip()
        blocks.append(block)
    return "\n\n".join(blocks)


def build_prompt(row, few_shot_examples):
    examples_text = format_few_shot_examples(few_shot_examples)

    return f"""
{SYSTEM_PROMPT}

Below are 5 labeled examples, one for each grading category.

{examples_text}

Now grade the following new student response.

Question:
{row['question']}

Reference answer:
{row['reference_answer']}

Student answer:
{row['student_answer']}
""".strip()


def make_batch_request(row, few_shot_examples):
    prompt = build_prompt(row, few_shot_examples)

    return {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": prompt,
            "max_output_tokens": 16,
        },
    }


# =========================
# Step 1: Load dataset
# =========================
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_df = dataset[TRAIN_SPLIT].to_pandas()
test_df = dataset[TEST_SPLIT].to_pandas()

if MAX_ROWS is not None:
    test_df = test_df.head(MAX_ROWS).copy()

print(f"Train rows: {len(train_df)}")
print(f"Test rows used: {len(test_df)}")

# =========================
# Step 2: Get fixed few-shot examples
# =========================
few_shot_examples = select_and_save_few_shot_examples(
    train_df=train_df,
    few_shot_path=FEW_SHOT_PATH,
    random_seed=RANDOM_SEED,
)

print("\nFew-shot examples being used:")
for ex in few_shot_examples:
    print(f"Label {ex['label']} ({LABEL_MAP[ex['label']]}), id={ex['id']}")

# =========================
# Step 3: Create JSONL
# =========================
print("\nBuilding batch JSONL...")
with open(BATCH_INPUT_PATH, "w", encoding="utf-8") as f:
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        req = make_batch_request(row, few_shot_examples)
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

# =========================
# Step 4: Create OpenAI client
# =========================
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Step 5: Upload batch file
# =========================
print("Uploading batch file...")
with open(BATCH_INPUT_PATH, "rb") as f:
    uploaded = client.files.create(file=f, purpose="batch")

print("File uploaded:", uploaded.id)

# =========================
# Step 6: Create batch job
# =========================
print("Creating batch job...")
batch = client.batches.create(
    input_file_id=uploaded.id,
    endpoint="/v1/responses",
    completion_window="24h",
)

print("Batch id:", batch.id)

# =========================
# Step 7: Poll job status
# =========================
terminal_states = {"completed", "failed", "cancelled", "expired"}

while True:
    batch = client.batches.retrieve(batch.id)
    print("Batch status:", batch.status)

    if batch.status in terminal_states:
        break

    time.sleep(POLL_SECONDS)

# =========================
# Step 8: Download results
# =========================
if batch.output_file_id:
    print("Downloading results...")
    content = client.files.content(batch.output_file_id)
    RAW_RESULTS_PATH.write_text(content.text, encoding="utf-8")
else:
    print("No output file produced")

# =========================
# Step 9: Download errors
# =========================
if batch.error_file_id:
    print("Downloading error file...")
    err = client.files.content(batch.error_file_id)
    RAW_ERRORS_PATH.write_text(err.text, encoding="utf-8")
    print("\nError file content:")
    print(err.text)

# =========================
# Step 10: Parse predictions
# =========================
pred_map = {}

if RAW_RESULTS_PATH.exists():
    with open(RAW_RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["custom_id"]

            try:
                text = obj["response"]["body"]["output"][0]["content"][0]["text"]
            except Exception:
                text = None

            pred_map[cid] = clean_model_label(text)

# =========================
# Step 11: Merge with dataframe
# =========================
test_df["llm_label"] = test_df["id"].astype(str).map(pred_map)
test_df["llm_matches_gold"] = test_df["llm_label"] == test_df["label"]

# =========================
# Step 12: Save results
# =========================
test_df.to_csv(OUTPUT_CSV_PATH, index=False)
print("Saved results to:", OUTPUT_CSV_PATH)

# =========================
# Step 13: Quick evaluation
# =========================
valid = test_df["llm_label"].notna().sum()
print("Predictions parsed:", valid)

if valid > 0:
    acc = test_df.loc[test_df["llm_label"].notna(), "llm_matches_gold"].mean()
    print("Accuracy:", acc)

# =========================
# Step 14: Full analysis
# =========================
y_true = test_df["label"]
y_pred = test_df["llm_label"]

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