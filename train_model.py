# train_model.py
"""
Trains LightGBM on dataset_features.csv.
Bias removal and architecture rebalancing is handled by build_dataset.py.

Run after: python build_dataset.py
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)
import lightgbm as lgb

DATASET_CSV = "dataset_features.csv"

print("=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────
print("\n[1/4] Loading dataset...")

if not __import__('pathlib').Path(DATASET_CSV).exists():
    print(f"\n  ERROR: {DATASET_CSV} not found.")
    print("  Run: python build_dataset.py first.")
    exit(1)

df = pd.read_csv(DATASET_CSV)
print(f"  Samples  : {len(df)}")
print(f"  Features : {len(df.columns) - 1}")
print(f"  Benign   : {(df['label'] == 0).sum()}")
print(f"  Malicious: {(df['label'] == 1).sum()}")
print(f"  Columns  : {list(df.columns)}")

# ── Guard: catch empty dataset before it crashes sklearn ──
if len(df) == 0:
    print("\n  ERROR: dataset_features.csv is empty.")
    print("  Delete it and re-run: python build_dataset.py")
    exit(1)

if len(df.columns) <= 1:
    print("\n  ERROR: No feature columns found — only label present.")
    print("  Delete dataset_features.csv and re-run: python build_dataset.py")
    exit(1)

if df.isnull().all(axis=None):
    print("\n  ERROR: All values are NaN.")
    print("  Delete dataset_features.csv and re-run: python build_dataset.py")
    exit(1)

X = df.drop('label', axis=1)

# Drop any columns that are entirely NaN or non-numeric
bad_cols = [c for c in X.columns
            if X[c].isnull().all() or not pd.api.types.is_numeric_dtype(X[c])]
if bad_cols:
    print(f"  Dropping {len(bad_cols)} bad columns: {bad_cols}")
    X = X.drop(columns=bad_cols)

X = X.fillna(0)
y = df['label']
feature_columns = list(X.columns)

print(f"  Final features used : {len(feature_columns)}")
print(f"  Feature sample      : {feature_columns[:5]}")

if len(feature_columns) == 0:
    print("\n  ERROR: No valid feature columns remain after cleaning.")
    print("  Re-run: python build_dataset.py")
    exit(1)

# ── 2. Split and scale ────────────────────────────────────
print("\n[2/4] Splitting and scaling...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feature_columns)

print(f"  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)} samples")

# ── 3. Train ───────────────────────────────────────────────
print("\n[3/4] Training LightGBM...")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)
model.fit(X_train_s, y_train)

# ── 4. Evaluate and save ───────────────────────────────────
print("\n[4/4] Evaluating and saving...")

y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

accuracy  = accuracy_score(y_test,  y_pred)
f1        = f1_score(y_test,        y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test,    y_pred)
roc_auc   = roc_auc_score(y_test,   y_proba)

print(f"\n  Accuracy  : {accuracy:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

cv_scores = cross_val_score(
    model, X_train_s, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1'
)
print(f"  CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

if accuracy > 0.99:
    print("\n  WARNING: Accuracy is suspiciously high (>99%).")
    print("  Re-run build_dataset.py to check for remaining bias.")

print("\n" + classification_report(
    y_test, y_pred, target_names=['Benign', 'Malicious']
))

feat_imp = pd.DataFrame({
    'feature':    feature_columns,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("  Top 10 features:")
for _, row in feat_imp.head(10).iterrows():
    print(f"    {row['feature']:<35} {row['importance']:.0f}")

# Save artifacts
joblib.dump(model,           "malware_model.pkl")
joblib.dump(scaler,          "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

fpr, tpr, _  = roc_curve(y_test, y_proba)
prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)

joblib.dump({
    "accuracy":   accuracy,
    "f1":         f1,
    "precision":  precision,
    "recall":     recall,
    "roc_auc":    roc_auc,
    "avg_prec":   average_precision_score(y_test, y_proba),
    "fpr":        fpr,
    "tpr":        tpr,
    "prec_curve": prec_c,
    "rec_curve":  rec_c,
    "cm":         confusion_matrix(y_test, y_pred),
    "cv_scores":  cv_scores,
    "feat_imp":   feat_imp,
    "y_test":     np.array(y_test),
    "y_pred":     y_pred,
    "y_proba":    y_proba,
}, "eval_results.pkl")

print("\n  Saved: malware_model.pkl, scaler.pkl,")
print("         feature_columns.pkl, eval_results.pkl")
print("\nNext: python visualize_results.py")