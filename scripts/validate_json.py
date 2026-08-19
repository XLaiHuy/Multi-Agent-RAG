#!/usr/bin/env python3
"""
Normal readable JSON validator script.
"""
import sys
import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    broken_files = []
    scanned_count = 0

    for file_path in repo_root.rglob("*.json"):
        # Exclude dependency and environment directories
        if "node_modules" in file_path.parts or ".venv" in file_path.parts or ".git" in file_path.parts:
            continue

        scanned_count += 1
        try:
            content = file_path.read_text(encoding="utf-8")
            json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError, Exception) as exc:
            broken_files.append((str(file_path), str(exc)))

    if broken_files:
        print(f"FAILED: Found {len(broken_files)} broken JSON files:")
        for path, err in broken_files:
            print(f"  - {path}: {err}")
        sys.exit(1)

    print(f"SUCCESS: All {scanned_count} tracked JSON files parsed successfully with zero errors.")
    sys.exit(0)


if __name__ == "__main__":
    main()
