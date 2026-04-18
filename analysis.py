import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# =========================================================
# CONFIGURATION
# =========================================================
RUN_FILES = {
    "gpt4o_zero_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-4o_with_llm_labels_zero_shot.csv",
    "gpt4o_five_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-4o_with_llm_labels_five_shot.csv",
    "gpt4o_fifteen_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-4o_with_llm_labels_fifteen_shot.csv",
    "gpt52_zero_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-5.2_with_llm_labels_zero_shot.csv",
    "gpt52_five_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-5.2_with_llm_labels_five_shot.csv",
    "gpt52_fifteen_shot": r"scientsbank_batch_run/scientsbank_test_ua_gpt-5.2_with_llm_labels_fifteen_shot.csv",
}

OUT_DIR = Path("analysis_results")
OUT_DIR.mkdir(exist_ok=True)

LABEL_NAMES = {
    0: "correct",
    1: "contradictory",
    2: "partially_correct_incomplete",
    3: "irrelevant",
    4: "non_domain",
}

LABEL_ORDER = [0, 1, 2, 3, 4]

# How many misclassified examples per true label to save
N_ERROR_EXAMPLES_PER_LABEL = 5


# =========================================================
# HELPERS
# =========================================================
def get_common_confusions(cm_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Return the most frequent off-diagonal confusions.
    """
    rows = []
    for true_label in cm_df.index:
        for pred_label in cm_df.columns:
            if true_label != pred_label:
                count = cm_df.loc[true_label, pred_label]
                rows.append(
                    {
                        "true_label_id": true_label,
                        "true_label_name": LABEL_NAMES[true_label],
                        "pred_label_id": pred_label,
                        "pred_label_name": LABEL_NAMES[pred_label],
                        "count": int(count),
                    }
                )

    confusion_df = pd.DataFrame(rows)
    confusion_df = confusion_df.sort_values(by="count", ascending=False).head(top_k)
    return confusion_df


def save_error_examples(eval_df: pd.DataFrame, run_name: str) -> None:
    """
    Save a few misclassified examples for each true label.
    """
    error_rows = []

    for label in LABEL_ORDER:
        subset = eval_df[
            (eval_df["label"] == label) &
            (eval_df["llm_label"] != eval_df["label"])
        ].copy()

        subset = subset.head(N_ERROR_EXAMPLES_PER_LABEL)

        for _, row in subset.iterrows():
            error_rows.append(
                {
                    "run": run_name,
                    "id": row.get("id", ""),
                    "true_label_id": int(row["label"]),
                    "true_label_name": LABEL_NAMES[int(row["label"])],
                    "pred_label_id": int(row["llm_label"]),
                    "pred_label_name": LABEL_NAMES[int(row["llm_label"])],
                    "question": row.get("question", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "student_answer": row.get("student_answer", ""),
                }
            )

    error_df = pd.DataFrame(error_rows)
    error_path = OUT_DIR / f"{run_name}_error_examples.csv"
    error_df.to_csv(error_path, index=False)


def analyze_run(run_name: str, csv_path: str):
    df = pd.read_csv(csv_path)

    if "label" not in df.columns or "llm_label" not in df.columns:
        raise ValueError(
            f"{run_name}: expected columns 'label' and 'llm_label' were not found."
        )

    # Keep only rows with parsed predictions
    eval_df = df[df["llm_label"].notna()].copy()
    eval_df["label"] = eval_df["label"].astype(int)
    eval_df["llm_label"] = eval_df["llm_label"].astype(int)

    y_true = eval_df["label"]
    y_pred = eval_df["llm_label"]

    # Overall metrics
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

    # Classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=[LABEL_NAMES[i] for i in LABEL_ORDER],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = OUT_DIR / f"{run_name}_classification_report.csv"
    report_df.to_csv(report_path, index=True)

    # Per-class summary table
    per_class_rows = []
    for label in LABEL_ORDER:
        label_name = LABEL_NAMES[label]
        metrics = report_dict[label_name]

        subset = eval_df[eval_df["label"] == label]
        per_label_accuracy = (
            (subset["label"] == subset["llm_label"]).mean() if len(subset) > 0 else None
        )

        per_class_rows.append(
            {
                "run": run_name,
                "label_id": label,
                "label_name": label_name,
                "count": len(subset),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": metrics["support"],
                "per_label_accuracy": per_label_accuracy,
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_path = OUT_DIR / f"{run_name}_per_class_summary.csv"
    per_class_df.to_csv(per_class_path, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=LABEL_ORDER,
        columns=LABEL_ORDER,
    )
    cm_path = OUT_DIR / f"{run_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_path, index=True)

    # Common confusions
    common_confusions_df = get_common_confusions(cm_df, top_k=10)
    common_confusions_path = OUT_DIR / f"{run_name}_top_confusions.csv"
    common_confusions_df.to_csv(common_confusions_path, index=False)

    # Save misclassified examples
    save_error_examples(eval_df, run_name)

    # Hardest / easiest classes by F1
    hardest_class = per_class_df.sort_values(by="f1_score", ascending=True).iloc[0]
    easiest_class = per_class_df.sort_values(by="f1_score", ascending=False).iloc[0]

    return (
        {
            "run": run_name,
            "file": csv_path,
            "total_rows_in_file": len(df),
            "evaluated_rows": len(eval_df),
            "accuracy": accuracy,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_p,
            "weighted_recall": weighted_r,
            "weighted_f1": weighted_f1,
            "hardest_label_by_f1": hardest_class["label_name"],
            "hardest_label_f1": hardest_class["f1_score"],
            "easiest_label_by_f1": easiest_class["label_name"],
            "easiest_label_f1": easiest_class["f1_score"],
        },
        per_class_df,
    )


# =========================================================
# MAIN
# =========================================================
def main():
    summary_rows = []
    all_per_class_rows = []

    for run_name, csv_path in RUN_FILES.items():
        path_obj = Path(csv_path)

        if not path_obj.exists():
            print(f"\nSkipping {run_name} — file not found: {csv_path}")
            continue

        print(f"\nAnalyzing: {run_name}")
        print(f"File: {csv_path}")

        metrics, per_class_df = analyze_run(run_name, csv_path)
        summary_rows.append(metrics)
        all_per_class_rows.append(per_class_df)

        print(f"  Evaluated rows:  {metrics['evaluated_rows']}")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
        print(
            f"  Hardest class:   {metrics['hardest_label_by_f1']} "
            f"(F1={metrics['hardest_label_f1']:.4f})"
        )
        print(
            f"  Easiest class:   {metrics['easiest_label_by_f1']} "
            f"(F1={metrics['easiest_label_f1']:.4f})"
        )

    if not summary_rows:
        print("\nNo files were analyzed. Please check the file paths.")
        return

    # Save all-runs summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="macro_f1", ascending=False)

    summary_path = OUT_DIR / "all_runs_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save combined per-class summary
    combined_per_class_df = pd.concat(all_per_class_rows, ignore_index=True)
    combined_per_class_path = OUT_DIR / "all_runs_per_class_summary.csv"
    combined_per_class_df.to_csv(combined_per_class_path, index=False)

    # Hardest categories across all runs
    hardest_overall = combined_per_class_df.sort_values(by="f1_score", ascending=True)
    hardest_overall_path = OUT_DIR / "hardest_categories_across_runs.csv"
    hardest_overall.to_csv(hardest_overall_path, index=False)

    print("\n=================================================")
    print("SUMMARY OF ALL RUNS")
    print("=================================================")
    print(summary_df.to_string(index=False))

    print("\n=================================================")
    print("HARDEST CATEGORIES ACROSS RUNS (lowest F1 first)")
    print("=================================================")
    print(
        hardest_overall[
            ["run", "label_name", "count", "precision", "recall", "f1_score"]
        ]
        .head(12)
        .to_string(index=False)
    )

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved combined per-class summary to: {combined_per_class_path}")
    print(f"Saved hardest-category table to: {hardest_overall_path}")
    print(f"Detailed reports saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()