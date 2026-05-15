# visualize_results.py
"""
Generates all evaluation plots from eval_results.pkl.
Output → plots/ directory (PNG, 300 DPI, paper-ready)

Run: python visualize_results.py
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

RESULTS_PKL = "eval_results.pkl"
OUTPUT_DIR  = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})
PALETTE = {"benign": "#2ecc71", "malicious": "#e74c3c", "main": "#3498db", "accent": "#9b59b6"}

# ── Load results ───────────────────────────────────────────
print("Loading eval_results.pkl...")
r = joblib.load(RESULTS_PKL)

# ── 1. Summary metrics bar chart ───────────────────────────
print("[1/6] Summary metrics...")
fig, ax = plt.subplots(figsize=(7, 4))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
values  = [r["accuracy"], r["precision"], r["recall"], r["f1"], r["roc_auc"]]
colors  = [PALETTE["main"]] * 4 + [PALETTE["accent"]]
bars = ax.bar(metrics, values, color=colors, width=0.5, zorder=3)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Performance Summary", fontweight="bold")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_metrics_summary.png")
plt.close()

# ── 2. Confusion matrix ────────────────────────────────────
print("[2/6] Confusion matrix...")
fig, ax = plt.subplots(figsize=(5, 4))
cm = r["cm"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"],
            linewidths=0.5, linecolor="white")
ax.set_xlabel("Predicted Label", fontweight="bold")
ax.set_ylabel("True Label", fontweight="bold")
ax.set_title("Confusion Matrix", fontweight="bold")
# Annotate TP / TN / FP / FN
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j + 0.5, i + 0.75, labels[i][j], ha="center",
                color="grey", fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_confusion_matrix.png")
plt.close()

# ── 3. ROC curve ───────────────────────────────────────────
print("[3/6] ROC curve...")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(r["fpr"], r["tpr"], color=PALETTE["main"], lw=2,
        label=f"ROC Curve (AUC = {r['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax.fill_between(r["fpr"], r["tpr"], alpha=0.08, color=PALETTE["main"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_roc_curve.png")
plt.close()

# ── 4. Precision-Recall curve ──────────────────────────────
print("[4/6] Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(r["rec_curve"], r["prec_curve"], color=PALETTE["accent"], lw=2,
        label=f"PR Curve (AP = {r['avg_prec']:.4f})")
ax.fill_between(r["rec_curve"], r["prec_curve"], alpha=0.08, color=PALETTE["accent"])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve", fontweight="bold")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_precision_recall_curve.png")
plt.close()

# ── 5. Cross-validation F1 scores ─────────────────────────
print("[5/6] Cross-validation scores...")
fig, ax = plt.subplots(figsize=(6, 4))
cv = r["cv_scores"]
folds = [f"Fold {i+1}" for i in range(len(cv))]
bars = ax.bar(folds, cv, color=PALETTE["main"], width=0.5, zorder=3)
ax.axhline(cv.mean(), color=PALETTE["malicious"], linestyle="--", lw=1.5,
           label=f"Mean F1 = {cv.mean():.4f}")
ax.fill_between([-0.5, len(cv) - 0.5],
                cv.mean() - cv.std(), cv.mean() + cv.std(),
                alpha=0.1, color=PALETTE["malicious"], label=f"±1 Std = {cv.std():.4f}")
for bar, val in zip(bars, cv):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(max(0, cv.min() - 0.05), 1.05)
ax.set_ylabel("F1 Score")
ax.set_title("5-Fold Cross-Validation F1 Scores", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_cross_validation.png")
plt.close()

# ── 6. Top 15 feature importances ─────────────────────────
print("[6/6] Feature importances...")
feat_imp = r["feat_imp"].head(15)
fig, ax = plt.subplots(figsize=(8, 5))
colors = [PALETTE["main"] if i < 5 else PALETTE["accent"] if i < 10
          else "#95a5a6" for i in range(len(feat_imp))]
ax.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1],
        color=colors[::-1], zorder=3)
ax.set_xlabel("Importance Score")
ax.set_title("Top 15 Feature Importances", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_feature_importance.png")
plt.close()

# ── Done ───────────────────────────────────────────────────
print(f"\nAll plots saved to → {OUTPUT_DIR}/")
print("  1_metrics_summary.png")
print("  2_confusion_matrix.png")
print("  3_roc_curve.png")
print("  4_precision_recall_curve.png")
print("  5_cross_validation.png")
print("  6_feature_importance.png")