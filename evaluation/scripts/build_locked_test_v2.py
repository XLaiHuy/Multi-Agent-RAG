#!/usr/bin/env python3
"""
Build LOCKED_TEST_V2 Split for Rigorous Large-Scale Evaluation.
Extracts 25 disjoint contracts (indices 40..64) from CUADv1.json.
Guarantees ZERO contract overlap with DEV (indices 15..34) and LEGACY_TEST_V1 (indices 3..12).
Generates >= 100 answerable queries + 50 unanswerable queries into evaluation/manifests/cuad_locked_test_v2_manifest.json.
"""
import os
import sys
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

RAW_JSON_PATH = Path("evaluation/datasets/cuad/raw/CUADv1.json")
CONTRACTS_DIR = Path("evaluation/datasets/cuad/processed/contracts")
MANIFEST_DIR = Path("evaluation/manifests")

sys.path.insert(0, r"C:\Users\HUY\.gemini\antigravity-ide\brain\b3fc2c97-4747-4204-8666-5a9ef508e9a8\scratch")
from build_dev_split import CLAUSE_QUERY_MAP

def build_locked_test_v2():
    print(f"[TEST V2 Builder] Loading {RAW_JSON_PATH}...")
    cuad_data = json.loads(RAW_JSON_PATH.read_text(encoding="utf-8"))
    contracts_raw = cuad_data.get("data", [])

    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Disjoint indices:
    dev_indices = set(range(15, 35))
    legacy_test_indices = set(range(3, 13))
    test_v2_indices = list(range(40, 65)) # 25 contracts

    assert len(set(test_v2_indices).intersection(dev_indices)) == 0, "ERROR: Overlap between TEST_V2 and DEV!"
    assert len(set(test_v2_indices).intersection(legacy_test_indices)) == 0, "ERROR: Overlap between TEST_V2 and LEGACY_TEST!"

    test_v2_contracts = []
    test_v2_queries = []

    for c_idx in test_v2_indices:
        c_raw = contracts_raw[c_idx]
        title = c_raw["title"]
        safe_id = f"cuad_contract_{c_idx:03d}_{re.sub(r'[^a-zA-Z0-9_]', '_', title)[:30]}"
        paragraph = c_raw["paragraphs"][0]
        context = paragraph["context"]

        c_md_file = CONTRACTS_DIR / f"{safe_id}.md"
        c_txt_file = CONTRACTS_DIR / f"{safe_id}.txt"
        if not c_md_file.exists():
            c_md_file.write_text(f"# {title}\n\n{context}", encoding="utf-8")
        if not c_txt_file.exists():
            c_txt_file.write_text(context, encoding="utf-8")

        qas = paragraph["qas"]
        valid_annotations = 0

        for q_idx, qa in enumerate(qas):
            raw_q = qa["question"]
            is_impossible = qa.get("is_impossible", False)
            answers = qa.get("answers", [])

            cat = "General"
            for known_cat in CLAUSE_QUERY_MAP:
                if known_cat.lower() in raw_q.lower():
                    cat = known_cat
                    break

            nl_question = CLAUSE_QUERY_MAP.get(cat, raw_q)

            if not is_impossible and len(answers) > 0:
                first_ans = answers[0]
                gold_text = first_ans["text"]
                gold_start = first_ans["answer_start"]
                valid_annotations += 1

                test_v2_queries.append({
                    "query_id": f"test_v2_cuad_{safe_id[:20]}_{cat.replace(' ', '_')}_{q_idx}",
                    "source_contract_id": safe_id,
                    "category": cat,
                    "question": nl_question,
                    "is_unanswerable": False,
                    "gold_evidence": gold_text,
                    "gold_answer_start": gold_start,
                    "gold_answers_count": len(answers),
                })
            elif is_impossible and q_idx % 2 == 0:
                test_v2_queries.append({
                    "query_id": f"test_v2_cuad_{safe_id[:20]}_unans_{cat.replace(' ', '_')}_{q_idx}",
                    "source_contract_id": safe_id,
                    "category": cat,
                    "question": nl_question,
                    "is_unanswerable": True,
                    "gold_evidence": "",
                    "gold_answer_start": -1,
                    "gold_answers_count": 0,
                })

        test_v2_contracts.append({
            "source_contract_id": safe_id,
            "original_title": title,
            "filename": f"{safe_id}.md",
            "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "char_length": len(context),
            "total_qas_in_source": len(qas),
            "valid_annotations_count": valid_annotations,
        })

    ans_count = sum(1 for q in test_v2_queries if not q["is_unanswerable"])
    unans_count = sum(1 for q in test_v2_queries if q["is_unanswerable"])

    manifest_data = {
        "dataset_name": "CUAD v1 LOCKED_TEST_V2 Split",
        "split": "LOCKED_TEST_V2",
        "total_contracts": len(test_v2_contracts),
        "total_queries": len(test_v2_queries),
        "total_answerable_queries": ans_count,
        "total_unanswerable_queries": unans_count,
        "contract_indices": test_v2_indices,
        "contracts": test_v2_contracts,
        "queries": test_v2_queries,
    }

    out_path = MANIFEST_DIR / "cuad_locked_test_v2_manifest.json"
    out_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"[OK] Wrote LOCKED_TEST_V2 manifest to {out_path}")
    print(f"     Contracts: {len(test_v2_contracts)}")
    print(f"     Total Queries: {len(test_v2_queries)} (Answerable: {ans_count}, Unanswerable: {unans_count})")

if __name__ == "__main__":
    build_locked_test_v2()
