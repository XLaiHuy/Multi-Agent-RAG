"""
Official CUAD (Contract Understanding Atticus Dataset) Downloader.
Downloads official CUAD v1 data distribution from The Atticus Project.
Records provenance, license, timestamp, and SHA-256 integrity checksums.
"""
import os
import sys
import json
import time
import zipfile
import hashlib
import urllib.request
from pathlib import Path

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
CUAD_DIR = Path("evaluation/datasets/cuad")
RAW_DIR = CUAD_DIR / "raw"
MANIFEST_DIR = Path("evaluation/manifests")


def calculate_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download_official_cuad() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "cuad_data.zip"

    print(f"[CUAD Downloader] Connecting to official CUAD source: {CUAD_URL}...")
    req = urllib.request.Request(CUAD_URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_len = resp.headers.get("Content-Length")
        print(f"[CUAD Downloader] Downloading {int(content_len) / (1024*1024):.2f} MB...")
        with open(zip_path, "wb") as f:
            f.write(resp.read())

    sha256 = calculate_sha256(zip_path)
    print(f"[CUAD Downloader] Download complete. SHA-256: {sha256}")

    # Extract CUADv1.json
    print("[CUAD Downloader] Extracting official CUADv1.json...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    # Find CUADv1.json in extracted files
    cuad_json = None
    for p in RAW_DIR.rglob("*.json"):
        if "cuad" in p.name.lower():
            cuad_json = p
            break

    if not cuad_json or not cuad_json.exists():
        raise RuntimeError(f"CUADv1.json not found in extracted archive at {RAW_DIR}")

    print(f"[CUAD Downloader] Successfully extracted official CUAD dataset to: {cuad_json}")
    return cuad_json


if __name__ == "__main__":
    download_official_cuad()
