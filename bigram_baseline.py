import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# =========================================================
# CONFIGURATION
# =========================================================
DATASET_NAME = "nkazi/SciEntsBank"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test_ua"

ADD_K = 0.01
MIN_COUNT = 2
MAX_ROWS = None  # use None for all rows, or e.g. 540

OUT_DIR = Path("bigram_baseline_results")
OUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV_PATH = OUT_DIR / f"bigram_baseline_{TEST_SPLIT}_predictions.csv"
SUMMARY_PATH = OUT_DIR / f"bigram_baseline_{TEST_SPLIT}_summary.txt"
CONFUSION_PATH = OUT_DIR / f"bigram_baseline_{TEST_SPLIT}_confusion_matrix.csv"
REPORT_PATH = OUT_DIR / f"bigram_baseline_{TEST_SPLIT}_classification_report.csv"

LABEL_NAMES = {
    0: "correct",
    1: "contradictory",
    2: "partially_correct_incomplete",
    3: "irrelevant",
    4: "non_domain",
}
LABEL_ORDER = [0, 1, 2, 3, 4]


# =========================================================
# TOKENIZATION
# =========================================================
def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    return ["<s>"] + tokens + ["</s>"]


def build_vocab(df_train: pd.DataFrame, min_count: int) -> set[str]:
    counts = Counter()

    for text in df_train["student_answer"].fillna(""):
        counts.update(tokenize(text))

    vocab = {"<s>", "</s>", "<unk>"}
    for w, c in counts.items():
        if w in {"<s>", "</s>"}:
            continue
        if c >= min_count:
            vocab.add(w)

    return vocab


def map_to_vocab(tokens: list[str], vocab: set[str]) -> list[str]:
    return [t if t in vocab else "<unk>" for t in tokens]


# =========================================================
# BIGRAM MODEL
# =========================================================
def build_bigram_counts(texts: list[str], vocab: set[str]):
    context_counts = Counter()
    bigram_counts = Counter()

    for text in texts:
        toks = map_to_vocab(tokenize(text), vocab)
        for h, w in zip(toks[:-1], toks[1:]):
            context_counts[h] += 1
            bigram_counts[(h, w)] += 1

    return context_counts, bigram_counts


def make_model(texts: list[str], vocab: set[str], add_k: float) -> dict:
    context_counts, bigram_counts = build_bigram_counts(texts, vocab)
    return {
        "context": context_counts,
        "bigram": bigram_counts,
        "V": len(vocab),
        "k": add_k,
        "vocab": vocab,
    }


def logprob(tokens: list[str], model: dict) -> float:
    context = model["context"]
    bigram = model["bigram"]
    V = model["V"]
    k = model["k"]

    s = 0.0
    for h, w in zip(tokens[:-1], tokens[1:]):
        num = bigram.get((h, w), 0) + k
        den = context.get(h, 0) + k * V
        s += math.log(num / den)
    return s


# =========================================================
# TRAIN
# =========================================================
def train_models(df_train: pd.DataFrame, vocab: set[str]) -> dict:
    models = {}

    for label in LABEL_ORDER:
        subset = df_train[df_train["label"] == label]
        texts = subset["student_answer"].fillna("").tolist()

        print(f"Training label {label} ({LABEL_NAMES[label]}): {len(texts)} examples")
        models[label] = make_model(texts, vocab, add_k=ADD_K)

    return models


# =========================================================
# PREDICT
# =========================================================
def predict_label(text: str, models: dict, vocab: set[str]) -> tuple[int, dict]:
    toks = map_to_vocab(tokenize(text), vocab)

    scores = {}
    for label, model in models.items():
        scores[label] = logprob(toks, model)

    pred = max(scores, key=scores.get)
    return pred, scores


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    df_train = dataset[TRAIN_SPLIT].to_pandas().copy()
    df_test = dataset[TEST_SPLIT].to_pandas().copy()

    if MAX_ROWS is not None:
        df_test = df_test.head(MAX_ROWS).copy()

    print(f"Train rows: {len(df_train)}")
    print(f"Test rows: {len(df_test)}")

    # Build vocabulary only from training split
    print("Building vocabulary...")
    vocab = build_vocab(df_train, min_count=MIN_COUNT)
    print(f"Vocabulary size: {len(vocab)}")

    # Train one model per class
    print("Training bigram models...")
    models = train_models(df_train, vocab)

    # Predict on test split
    print("Predicting on test set...")
    preds = []
    score_rows = []

    for _, row in df_test.iterrows():
        student_answer = row.get("student_answer", "")
        pred, scores = predict_label(student_answer, models, vocab)

        preds.append(pred)

        score_row = {"id": row.get("id", ""), "true_label": row["label"], "pred_label": pred}
        for label in LABEL_ORDER:
            score_row[f"logprob_{label}_{LABEL_NAMES[label]}"] = scores[label]
        score_rows.append(score_row)

    df_test["bigram_label"] = preds
    df_test["bigram_matches_gold"] = df_test["bigram_label"] == df_test["label"]

    # Save predictions
    df_test.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved predictions to: {OUTPUT_CSV_PATH}")

    # Metrics
    y_true = df_test["label"].astype(int)
    y_pred = df_test["bigram_label"].astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=[LABEL_NAMES[i] for i in LABEL_ORDER],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(REPORT_PATH)

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{LABEL_NAMES[i]}" for i in LABEL_ORDER],
        columns=[f"pred_{LABEL_NAMES[i]}" for i in LABEL_ORDER],
    )
    cm_df.to_csv(CONFUSION_PATH)

    # Print summary
    summary = []
    summary.append(f"Train split: {TRAIN_SPLIT}")
    summary.append(f"Test split: {TEST_SPLIT}")
    summary.append(f"Train rows: {len(df_train)}")
    summary.append(f"Test rows: {len(df_test)}")
    summary.append(f"Vocabulary size: {len(vocab)}")
    summary.append(f"ADD_K: {ADD_K}")
    summary.append(f"MIN_COUNT: {MIN_COUNT}")
    summary.append("")
    summary.append(f"Accuracy: {accuracy:.4f}")
    summary.append(f"Macro Precision: {macro_p:.4f}")
    summary.append(f"Macro Recall: {macro_r:.4f}")
    summary.append(f"Macro F1: {macro_f1:.4f}")
    summary.append(f"Weighted Precision: {weighted_p:.4f}")
    summary.append(f"Weighted Recall: {weighted_r:.4f}")
    summary.append(f"Weighted F1: {weighted_f1:.4f}")
    summary.append("")
    summary.append("Classification Report:")
    summary.append(report_df.to_string())
    summary.append("")
    summary.append("Confusion Matrix:")
    summary.append(cm_df.to_string())

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved classification report to: {REPORT_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_PATH}")


if __name__ == "__main__":
    main()