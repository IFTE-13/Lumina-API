import pandas as pd
from pathlib import Path

csv = Path("dataset_features.csv")
if not csv.exists():
    print("dataset_features.csv does not exist — run build_dataset.py")
    exit()

df = pd.read_csv(csv)
print(f"Shape       : {df.shape}")
print(f"Features    : {len(df.columns) - 1}")
print(f"Columns     : {list(df.columns[:10])}...")  # Show first 10
print(f"Benign      : {(df['label']==0).sum() if 'label' in df.columns else 'NO LABEL'}")
print(f"Malicious   : {(df['label']==1).sum() if 'label' in df.columns else 'NO LABEL'}")
print(f"NaN count   : {df.isnull().sum().sum()}")

if len(df.columns) > 1:
    print(f"\nFirst row features (first 5):")
    first_row = df.iloc[0]
    for col in df.columns[:5]:
        if col != 'label':
            print(f"  {col}: {first_row[col]}")
else:
    print("\nERROR: No feature columns found!")