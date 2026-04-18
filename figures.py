import matplotlib.pyplot as plt

# Data
models = [
    "Bigram\nBaseline",
    "GPT-5.2\n(5-shot)",
    "GPT-4o\n(5-shot)",
    "JMD-ASAG\n(2019)",
    "BERT-base\n(2020)",
    "MDA-ASAS\n(2020)"
]

scores = [0.542, 0.590, 0.653, 0.657, 0.662, 0.700]

# Create figure
plt.figure(figsize=(10, 5))

bars = plt.bar(models, scores)

# Title and labels
plt.title("Comparison of Models on SciEntsBank (test_ua)", fontsize=13)
plt.ylabel("Weighted F1 Score", fontsize=11)

# Y-axis formatting
plt.ylim(0, 0.75)
plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# Improve x-label readability
plt.xticks(rotation=25, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.3f}",
        ha='center',
        va='bottom',
        fontsize=9
    )

# Remove unnecessary borders
for spine in ["top", "right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("model_comparison.eps", format="eps", bbox_inches="tight")
plt.show()





import matplotlib.pyplot as plt
import numpy as np

# Apply style
plt.style.use("seaborn-v0_8")

# Data
classes = [
    "Correct",
    "Contradictory",
    "Partially\nCorrect",
    "Irrelevant",
    "Non-domain"
]

gpt4o = [0.802, 0.645, 0.422, 0.608, 0.000]
gpt52 = [0.797, 0.672, 0.461, 0.305, 0.364]

x = np.arange(len(classes))
width = 0.35

plt.figure(figsize=(10, 5))

bars1 = plt.bar(x - width/2, gpt4o, width, label="GPT-4o (5-shot)")
bars2 = plt.bar(x + width/2, gpt52, width, label="GPT-5.2 (5-shot)")

plt.title("Per-Class Performance Comparison", fontsize=13)
plt.ylabel("F1 Score")

plt.xticks(x, classes)
plt.ylim(0, 0.9)

plt.legend(frameon=False)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

plt.tight_layout()
plt.savefig("per_class_comparison.eps", format="eps", bbox_inches="tight")
plt.show()











# =========================================
# CONFUSION MATRIX PLOTS
# =========================================

import numpy as np

# Labels
labels = ["Correct", "Contradictory", "Partially", "Irrelevant", "Non-domain"]

# Confusion matrices
cm_gpt4o = np.array([
    [194, 1, 34, 4, 0],
    [0, 40, 11, 7, 0],
    [50, 5, 53, 5, 0],
    [7, 20, 39, 66, 1],
    [0, 0, 1, 2, 0]
])

cm_gpt52 = np.array([
    [175, 0, 57, 1, 0],
    [0, 40, 15, 3, 0],
    [29, 2, 80, 2, 0],
    [2, 19, 81, 25, 6],
    [0, 0, 1, 0, 2]
])


def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))

    im = plt.imshow(cm, cmap="Blues")

    plt.title(title, fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha='right')
    plt.yticks(np.arange(len(labels)), labels)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                j, i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color=color
            )

    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, format="eps", bbox_inches="tight")
    plt.show()


def plot_normalized_confusion_matrix(cm, title, filename):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))

    im = plt.imshow(cm_norm, cmap="Blues")

    plt.title(title + " (Normalized)", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha='right')
    plt.yticks(np.arange(len(labels)), labels)

    plt.colorbar(im, fraction=0.046, pad=0.04)

    threshold = cm_norm.max() / 2

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = cm_norm[i, j]
            color = "white" if value > threshold else "black"
            plt.text(
                j, i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color
            )

    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, format="eps", bbox_inches="tight")
    plt.show()


# ---- Generate confusion matrices ----

# Raw counts
plot_confusion_matrix(
    cm_gpt4o,
    "Confusion Matrix - GPT-4o (5-shot)",
    "cm_gpt4o_5shot.eps"
)

plot_confusion_matrix(
    cm_gpt52,
    "Confusion Matrix - GPT-5.2 (5-shot)",
    "cm_gpt52_5shot.eps"
)

# Normalized versions
plot_normalized_confusion_matrix(
    cm_gpt4o,
    "Confusion Matrix - GPT-4o (5-shot)",
    "cm_gpt4o_5shot_normalized.eps"
)

plot_normalized_confusion_matrix(
    cm_gpt52,
    "Confusion Matrix - GPT-5.2 (5-shot)",
    "cm_gpt52_5shot_normalized.eps"
)