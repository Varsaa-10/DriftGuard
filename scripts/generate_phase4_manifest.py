import hashlib
import json
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "archive" / "phase4_source_sha256.txt"
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

hashes = {}
for file_path in SRC_ROOT.rglob("*.*"):
    if file_path.is_file():
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        rel = file_path.relative_to(SRC_ROOT)
        hashes[str(rel)] = h.hexdigest()

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(hashes, f, indent=2, sort_keys=True)
print(f"SHA-256 manifest written to {MANIFEST_PATH}")
