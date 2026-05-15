from pathlib import Path
from feature_extractor import extract_features

benign_dir = Path("dataset/benign")
exe_files = list(benign_dir.glob("*.exe"))

print(f"Found {len(exe_files)} benign .exe files")

if exe_files:
    # Test extract features from first file
    features = extract_features(str(exe_files[0]))
    if features:
        print(f"\nFeatures extracted from first benign file:")
        print(f"Number of features: {len(features)}")
        print(f"First 10 features: {list(features.keys())[:10]}")
        print(f"Feature names: {list(features.keys())}")
    else:
        print("ERROR: Could not extract features from benign file!")
        print("Check if feature_extractor.py is working correctly")
else:
    print("ERROR: No .exe files found in dataset/benign/")
    print("Run: python setup_benign_data.py")