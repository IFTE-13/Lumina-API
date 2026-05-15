# feature_extractor.py
"""
Extracts exactly the same features as Mendeley's Header.csv.
This guarantees training and prediction use identical features.

Dataset: https://data.mendeley.com/datasets/vnj7sxkt53/1
"""
import pefile
import os
import math
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import traceback

SUSPICIOUS_IMPORTS = {
    "virtualalloc", "virtualprotect", "createremotethread",
    "writeprocessmemory", "readprocessmemory", "createprocess",
    "shellexecute", "winexec", "loadlibrary", "getprocaddress",
    "internetopen", "internetconnect", "httpsendrequesta",
    "regsetvalue", "regcreatekey", "regdeletekey",
    "cryptencrypt", "cryptdecrypt", "isdebuggerpresent",
    "checkremotedebuggerpresent", "outputdebugstring",
}

SUSPICIOUS_SECTION_NAMES = {
    b"upx0", b"upx1", b"upx2",
    b".aspack", b"aspack",
    b".adata",
    b"pec2",
    b".nsp0", b".nsp1",
    b"themida",
}


def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    entropy = 0.0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += -p_x * math.log2(p_x)
    return entropy


def extract_features(file_path: str) -> Optional[Dict]:
    """
    Extracts features matching Mendeley Header.csv column layout exactly.
    Reads entire file into memory first to avoid file handle conflicts.
    """
    try:
        abs_path = os.path.abspath(file_path)

        if not os.path.exists(abs_path):
            print(f"  [skip] Not found: {abs_path}")
            return None

        if os.path.getsize(abs_path) == 0:
            print(f"  [skip] Empty file")
            return None
        
        file_size = os.path.getsize(abs_path)
        if file_size == 0:
            print(f"  [skip] Empty file")
            return None
        
        if file_size < 1024:  # Less than 1KB is suspicious
            print(f"  [skip] File too small ({file_size} bytes)")
            return None

        # Read entire file into memory first — avoids all file handle conflicts
        try:
            with open(abs_path, 'rb') as f:
                raw = f.read()
        except PermissionError:
            print(f"  [skip] Permission denied: {Path(abs_path).name}")
            return None
        except Exception as e:
            print(f"  [skip] Cannot read: {e}")
            return None

        if len(raw) < 2 or raw[:2] != b'MZ':
            print(f"  [skip] Not a PE file (no MZ header)")
            return None

        try:
            pe = pefile.PE(data=raw)
        except pefile.PEFormatError as e:
            print(f"  [skip] PE format error: {e}")
            return None
        except Exception as e:
            print(f"  [skip] PE parsing error: {e}")
            return None

        features: Dict = {}

        # ── DOS Header ────────────────────────────────────
        dh = pe.DOS_HEADER
        features['e_magic']    = getattr(dh, 'e_magic',    0)
        features['e_cblp']     = getattr(dh, 'e_cblp',     0)
        features['e_cp']       = getattr(dh, 'e_cp',       0)
        features['e_crlc']     = getattr(dh, 'e_crlc',     0)
        features['e_cparhdr']  = getattr(dh, 'e_cparhdr',  0)
        features['e_minalloc'] = getattr(dh, 'e_minalloc', 0)
        features['e_maxalloc'] = getattr(dh, 'e_maxalloc', 0)
        features['e_ss']       = getattr(dh, 'e_ss',       0)
        features['e_sp']       = getattr(dh, 'e_sp',       0)
        features['e_csum']     = getattr(dh, 'e_csum',     0)
        features['e_ip']       = getattr(dh, 'e_ip',       0)
        features['e_cs']       = getattr(dh, 'e_cs',       0)
        features['e_lfarlc']   = getattr(dh, 'e_lfarlc',   0)
        features['e_ovno']     = getattr(dh, 'e_ovno',     0)
        features['e_oemid']    = getattr(dh, 'e_oemid',    0)
        features['e_oeminfo']  = getattr(dh, 'e_oeminfo',  0)
        features['e_lfanew']   = getattr(dh, 'e_lfanew',   0)

        # ── File Header ───────────────────────────────────
        fh = pe.FILE_HEADER
        features['Machine']              = getattr(fh, 'Machine',              0)
        features['NumberOfSections']     = getattr(fh, 'NumberOfSections',     0)
        features['TimeDateStamp']        = getattr(fh, 'TimeDateStamp',        0)
        features['PointerToSymbolTable'] = getattr(fh, 'PointerToSymbolTable', 0)
        features['NumberOfSymbols']      = getattr(fh, 'NumberOfSymbols',      0)
        features['SizeOfOptionalHeader'] = getattr(fh, 'SizeOfOptionalHeader', 0)
        features['Characteristics']      = getattr(fh, 'Characteristics',      0)

        # ── Optional Header ───────────────────────────────
        oh = pe.OPTIONAL_HEADER
        features['Magic']                       = getattr(oh, 'Magic',                       0)
        features['MajorLinkerVersion']          = getattr(oh, 'MajorLinkerVersion',          0)
        features['MinorLinkerVersion']          = getattr(oh, 'MinorLinkerVersion',          0)
        features['SizeOfCode']                  = getattr(oh, 'SizeOfCode',                  0)
        features['SizeOfInitializedData']       = getattr(oh, 'SizeOfInitializedData',       0)
        features['SizeOfUninitializedData']     = getattr(oh, 'SizeOfUninitializedData',     0)
        features['AddressOfEntryPoint']         = getattr(oh, 'AddressOfEntryPoint',         0)
        features['BaseOfCode']                  = getattr(oh, 'BaseOfCode',                  0)
        features['ImageBase']                   = getattr(oh, 'ImageBase',                   0)
        features['SectionAlignment']            = getattr(oh, 'SectionAlignment',            0)
        features['FileAlignment']               = getattr(oh, 'FileAlignment',               0)
        features['MajorOperatingSystemVersion'] = getattr(oh, 'MajorOperatingSystemVersion', 0)
        features['MinorOperatingSystemVersion'] = getattr(oh, 'MinorOperatingSystemVersion', 0)
        features['MajorImageVersion']           = getattr(oh, 'MajorImageVersion',           0)
        features['MinorImageVersion']           = getattr(oh, 'MinorImageVersion',           0)
        features['MajorSubsystemVersion']       = getattr(oh, 'MajorSubsystemVersion',       0)
        features['MinorSubsystemVersion']       = getattr(oh, 'MinorSubsystemVersion',       0)
        features['Reserved1']                   = getattr(oh, 'Reserved1',                   0)  # ← Mendeley has this
        features['SizeOfImage']                 = getattr(oh, 'SizeOfImage',                 0)
        features['SizeOfHeaders']               = getattr(oh, 'SizeOfHeaders',               0)
        features['CheckSum']                    = getattr(oh, 'CheckSum',                    0)
        features['Subsystem']                   = getattr(oh, 'Subsystem',                   0)
        features['DllCharacteristics']          = getattr(oh, 'DllCharacteristics',          0)
        features['SizeOfStackReserve']          = getattr(oh, 'SizeOfStackReserve',          0)
        # NOTE: SizeOfStackCommit is NOT in Mendeley CSV — intentionally omitted
        features['SizeOfHeapReserve']           = getattr(oh, 'SizeOfHeapReserve',           0)
        features['SizeOfHeapCommit']            = getattr(oh, 'SizeOfHeapCommit',            0)
        features['LoaderFlags']                 = getattr(oh, 'LoaderFlags',                 0)
        features['NumberOfRvaAndSizes']         = getattr(oh, 'NumberOfRvaAndSizes',         0)

        pe.close()
        return features

    except Exception as e:
        print(f"  [error] {Path(file_path).name}: {e}")
        return None


def extract_features_from_folder(
    folder_path: str,
    label: int,
    limit: int = 2500,
) -> pd.DataFrame:
    folder    = Path(folder_path)
    exe_files = list(folder.rglob("*.exe"))[:limit]
    print(f"  Found {len(exe_files)} files in {folder_path}")

    rows = []
    for i, exe in enumerate(exe_files, 1):
        features = extract_features(str(exe))
        if features:
            features['label'] = label
            rows.append(features)
        if i % 100 == 0:
            print(f"  Processed {i}/{len(exe_files)} — valid: {len(rows)}")

    df = pd.DataFrame(rows).fillna(0)
    print(f"  Done: {len(df)} valid samples (label={label})")
    return df