import pandas as pd
from pathlib import Path

malicious_csv = Path("dataset/Header.csv")
if malicious_csv.exists():
    df = pd.read_csv(malicious_csv)
    print(f"Malicious CSV shape: {df.shape}")
    print(f"Columns (first 20): {list(df.columns[:20])}")
    print(f"Total columns: {len(df.columns)}")
else:
    print(f"ERROR: {malicious_csv} not found!")
    print("Please download Header.csv from:")
    print("https://data.mendeley.com/datasets/vnj7sxkt53/1")