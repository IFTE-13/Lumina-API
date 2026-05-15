"""
Builds dataset_features.csv from:
  - Benign   : dataset/benign/*.exe
  - Malicious: dataset/Header.csv (Mendeley PE Header dataset)
"""
import pandas as pd
from pathlib import Path
from feature_extractor import extract_features

MALICIOUS_CSV = Path("dataset/Header.csv")
BENIGN_DIR    = Path("dataset/benign")
OUTPUT_CSV    = Path("dataset_features.csv")
LIMIT         = 2500

print("=" * 60)
print("BUILDING DATASET")
print("=" * 60)

# ── Validate inputs ────────────────────────────────────────
if not MALICIOUS_CSV.exists():
    print(f"\n  ERROR: {MALICIOUS_CSV} not found.")
    print("  1. Go to https://data.mendeley.com/datasets/vnj7sxkt53/1")
    print("  2. Download Header.csv")
    print("  3. Place it at dataset/Header.csv")
    exit(1)

if not BENIGN_DIR.exists() or not list(BENIGN_DIR.glob("*.exe")):
    print(f"\n  ERROR: No .exe files found in {BENIGN_DIR}")
    print("  Run: python setup_benign_data.py")
    exit(1)

# ── 1. Load malicious from Mendeley CSV ───────────────────
print(f"\n[1/5] Loading malicious samples from Mendeley CSV...")

df_mal = pd.read_csv(MALICIOUS_CSV)
print(f"  Raw shape : {df_mal.shape}")

# Drop non-feature columns (SHA256 and Malware_Type)
non_feature_cols = ['SHA256', 'Malware_Type']
df_mal = df_mal.drop(columns=[c for c in non_feature_cols if c in df_mal.columns])
df_mal['label'] = 1

# Convert all to numeric
for col in df_mal.columns:
    if col != 'label':
        df_mal[col] = pd.to_numeric(df_mal[col], errors='coerce')
df_mal = df_mal.fillna(0)

print(f"  Malicious samples : {len(df_mal)}")
print(f"  Features          : {len(df_mal.columns) - 1}")

# ── 2. Extract benign from system .exe files ──────────────
print(f"\n[2/5] Extracting benign features from {BENIGN_DIR}...")

exe_files   = list(BENIGN_DIR.glob("*.exe"))
benign_rows = []
failed      = 0

print(f"  Found {len(exe_files)} .exe files")

for i, exe in enumerate(exe_files, 1):
    features = extract_features(str(exe))
    if features:
        features['label'] = 0
        benign_rows.append(features)
    else:
        failed += 1
    if i % 500 == 0 or i == len(exe_files):
        print(f"  Progress : {i}/{len(exe_files)} — valid: {len(benign_rows)}, failed: {failed}")

if len(benign_rows) < 50:
    print(f"\n  ERROR: Only {len(benign_rows)} benign samples extracted.")
    print("  Run: python setup_benign_data.py")
    exit(1)

df_ben = pd.DataFrame(benign_rows).fillna(0)
print(f"  Benign samples : {len(df_ben)}")
print(f"  Benign features: {len(df_ben.columns) - 1}")

# ── 3. Align columns (CRITICAL FIX) ──────────────────────
print("\n[3/5] Aligning columns...")

# Get feature columns (excluding 'label')
benign_features = [c for c in df_ben.columns if c != 'label']
malicious_features = [c for c in df_mal.columns if c != 'label']

print(f"  Benign features count    : {len(benign_features)}")
print(f"  Malicious features count : {len(malicious_features)}")

# Find common columns
common_cols = sorted(set(benign_features) & set(malicious_features))
print(f"  Common features count    : {len(common_cols)}")

if len(common_cols) < 10:
    print("\n  ERROR: Too few common columns!")
    print(f"  First 10 benign features: {benign_features[:10]}")
    print(f"  First 10 malicious features: {malicious_features[:10]}")
    exit(1)

print(f"  First 10 common features: {common_cols[:10]}")

# Keep only common columns + label
keep_cols = common_cols + ['label']
df_ben_aln = df_ben[keep_cols].copy()
df_mal_aln = df_mal[keep_cols].copy()

print(f"  Benign aligned shape    : {df_ben_aln.shape}")
print(f"  Malicious aligned shape : {df_mal_aln.shape}")

# ── 4. Remove biased features ───────────────────────────────
print("\n[4/5] Removing biased features...")

# Drop known biased features
BIASED_FEATURES = [
    'ImageBase',      # Architecture-correlated
    'Machine',        # Architecture indicator
    'TimeDateStamp',  # Collection time artifact
    'CheckSum',       # Often zero in malware, valid in benign
]

biased_to_drop = [f for f in BIASED_FEATURES if f in keep_cols]
if biased_to_drop:
    keep_cols = [c for c in keep_cols if c not in biased_to_drop]
    df_ben_aln = df_ben_aln[keep_cols]
    df_mal_aln = df_mal_aln[keep_cols]
    print(f"  Dropped biased features: {biased_to_drop}")

print(f"  Features after bias removal: {len(keep_cols) - 1}")

# ── 5. Balance classes by architecture ──────────────────────
print("\n[5/5] Balancing dataset...")

# Limit samples per class to avoid memory issues
n_samples = min(len(df_ben_aln), len(df_mal_aln), LIMIT)
print(f"  Taking {n_samples} samples per class")

# Sample equally from both classes
df_ben_sample = df_ben_aln.sample(n=n_samples, random_state=42)
df_mal_sample = df_mal_aln.sample(n=n_samples, random_state=42)

# Combine and shuffle
df_combined = pd.concat([df_ben_sample, df_mal_sample])
df_final = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Final cleanup - ensure all numeric
for col in df_final.columns:
    if col != 'label':
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
df_final = df_final.fillna(0)

# Remove constant columns (all same value)
constant_cols = [c for c in df_final.columns 
                 if c != 'label' and df_final[c].nunique() <= 1]
if constant_cols:
    df_final = df_final.drop(columns=constant_cols)
    print(f"  Dropped {len(constant_cols)} constant columns: {constant_cols[:5]}")

# Save to CSV
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'=' * 60}")
print(f"  Saved          → {OUTPUT_CSV}")
print(f"  Total samples  : {len(df_final)}")
print(f"  Benign   (0)   : {(df_final['label'] == 0).sum()}")
print(f"  Malicious (1)  : {(df_final['label'] == 1).sum()}")
print(f"  Features       : {len(df_final.columns) - 1}")
print(f"  Feature sample : {[c for c in df_final.columns if c != 'label'][:10]}")
print(f"{'=' * 60}")
print("\nNext: python check_dataset.py")
print("Then: python train_model.py")