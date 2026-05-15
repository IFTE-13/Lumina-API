"""
Copies Windows system .exe files into dataset/benign/.
These are trusted files already on your machine — no download needed.
Collects from System32 (64-bit) and SysWOW64 (32-bit) to match
the architecture mix in the Mendeley malicious dataset.

Run: python setup_benign_data.py
"""
import shutil
from pathlib import Path

OUTPUT_DIR = Path("dataset/benign")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    Path("C:/Windows/SysWOW64"),   # 32-bit first — matches malware arch
    Path("C:/Windows/System32"),   # 64-bit second
]

TARGET    = 500
collected = 0

print("=" * 50)
print("COLLECTING BENIGN SAMPLES")
print("=" * 50)

for search_dir in SEARCH_DIRS:
    if not search_dir.exists():
        print(f"\n  [skip] {search_dir} not found")
        continue

    exe_files = list(search_dir.glob("*.exe"))
    print(f"\n  {search_dir.name}: {len(exe_files)} .exe files found")

    for exe in exe_files:
        if collected >= TARGET:
            break
        try:
            dest = OUTPUT_DIR / exe.name
            if dest.exists():
                continue
            shutil.copy2(exe, dest)
            collected += 1
            if collected % 100 == 0:
                print(f"  Collected: {collected}/{TARGET}")
        except Exception as e:
            pass   # silently skip locked/permission-denied files

    if collected >= TARGET:
        break

print(f"\n  Total collected : {collected} benign files → {OUTPUT_DIR}")

if collected < 100:
    print("\n  WARNING: Less than 100 files collected.")
    print("  Check that System32 / SysWOW64 exist and are readable.")

print("\nNext: python build_dataset.py")