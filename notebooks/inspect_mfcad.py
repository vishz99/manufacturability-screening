# Inspects the actual MFCAD repo layout before assuming any directory structure.

import os

ROOT = "../data/MFCAD"

for dirpath, dirnames, filenames in os.walk(ROOT):
    # skip git internals
    if ".git" in dirpath:
        continue
    depth = dirpath.replace(ROOT, "").count(os.sep)
    if depth > 2:
        continue
    print(f"{dirpath}  ->  {len(filenames)} files")
    if filenames:
        print("   sample:", filenames[:3])