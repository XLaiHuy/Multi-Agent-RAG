#!/usr/bin/env python3
"""
Build Large CUAD DEV Split for Retrieval Optimization
Extracts 20 disjoint contracts (indices 15..34) from CUADv1.json.
Generates 100+ answerable queries + 50 unanswerable queries into evaluation/manifests/cuad_dev_manifest.json.
Guarantees ZERO contract leakage with the TEST set (indices 3..12).
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

# Natural language query templates for 41 Atticus clause categories
CLAUSE_QUERY_MAP = {
    "Document Name": "What is the official title and full document name of this agreement?",
    "Parties": "Who are the named parties entering into this agreement?",
    "Agreement Date": "What is the effective execution date of this agreement?",
    "Effective Date": "What is the designated effective start date specified in the agreement?",
    "Expiration Date": "When does the initial term of this agreement expire or terminate?",
    "Renewal Term": "What are the renewal terms, auto-renewal mechanisms, or extension conditions?",
    "Notice Period To Terminate Renewal": "What is the notice period required to cancel or prevent automatic renewal?",
    "Governing Law": "Which jurisdiction or state laws govern the interpretation of this agreement?",
    "Termination For Convenience": "Does the contract permit either party to terminate for convenience without cause, and what notice is required?",
    "Cap On Liability": "What is the maximum aggregate liability cap specified under the contract?",
    "Uncapped Liability": "Are there any uncapped liability exceptions or indemnities excluded from limitation of liability?",
    "Indemnification": "What indemnification and defense obligations are provided under the agreement?",
    "Non-Compete": "Does the contract impose non-compete restrictions on any party, and what is the scope/duration?",
    "Exclusivity": "Does the contract grant exclusive rights or impose exclusivity requirements?",
    "No-Solicit Of Customers": "Are there non-solicitation restrictions preventing solicitation of customers or clients?",
    "No-Solicit Of Employees": "Are there restrictions preventing the recruitment or solicitation of employees?",
    "Confidentiality / Non-Disclosure": "What confidentiality obligations and disclosure restrictions apply to proprietary information?",
    "Survival Of Obligations": "Which obligations survive the termination or expiration of this agreement?",
    "Audit Rights": "What audit, books, and inspection rights are granted, and what notice is required?",
    "Insurance": "What commercial general liability or professional insurance coverage must be maintained?",
    "Price Restriction": "Are there price restrictions, most favored nation pricing, or minimum price obligations?",
    "Minimum Commitment": "Are there minimum purchase volume commitments or guaranteed spending obligations?",
    "Anti-Assignment": "Can either party assign or transfer this agreement without prior written consent?",
    "Change Of Control": "Does a change of control or merger trigger termination rights or notification requirements?",
    "Force Majeure": "What events qualify as force majeure excusing performance under the agreement?",
    "Dispute Resolution": "What dispute resolution, mediation, or arbitration procedures are mandated?",
    "Warranty Duration": "What is the warranty period and what express warranties are provided?",
    "IP Ownership Assignment": "How does the contract allocate ownership of intellectual property and work product created?",
    "Post-Termination Services": "What post-termination transition or wind-down services must be provided?",
    "Revenue/Profit Sharing": "What revenue sharing, profit split, or royalty percentage terms apply?",
    "Volume Restriction": "Are there volume restrictions or maximum quantity thresholds?",
    "Covenant Not To Sue": "Does either party agree to a covenant not to sue or release of claims?",
    "Right Of First Refusal": "Is there a right of first refusal, first offer, or first negotiation granted?",
}

def clean_clause_category(q_str: str) -> str:
    cleaned = re.sub(r"^(Detail|Mention|Highlight|Identify|Locate|Find)\s+", "", q_str)
    cleaned = re.sub(r'[\'"]', '', cleaned).strip()
    return cleaned

def build_dev_split():
    print(f"[DEV Builder] Loading {RAW_JSON_PATH}...")
    cuad_data = json.loads(RAW_JSON_PATH.read_text(encoding="utf-8"))
    contracts_raw = cuad_data.get("data", [])
    print(f"[DEV Builder] Total contracts available: {len(contracts_raw)}")

    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Use indices 15 through 34 (20 contracts) for DEV
    dev_indices = list(range(15, 35))
    test_indices = list(range(3, 13))

    assert len(set(dev_indices).intersection(set(test_indices))) == 0, "ERROR: Overlap between DEV and TEST!"

    dev_contracts = []
    dev_queries = []

    for c_idx in dev_indices:
        c_raw = contracts_raw[c_idx]
        title = c_raw["title"]
        safe_id = f"cuad_contract_{c_idx:03d}_{re.sub(r'[^a-zA-Z0-9_]', '_', title)[:30]}"
        paragraph = c_raw["paragraphs"][0]
        context = paragraph["context"]

        # Write contract markdown
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

            # Extract category
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

                dev_queries.append({
                    "query_id": f"dev_cuad_{safe_id[:20]}_{cat.replace(' ', '_')}_{q_idx}",
                    "source_contract_id": safe_id,
                    "category": cat,
                    "question": nl_question,
                    "is_unanswerable": False,
                    "gold_evidence": gold_text,
                    "gold_answer_start": gold_start,
                    "gold_answers_count": len(answers),
                })
            elif is_impossible and q_idx % 2 == 0:  # sample unanswerables
                dev_queries.append({
                    "query_id": f"dev_cuad_{safe_id[:20]}_unans_{cat.replace(' ', '_')}_{q_idx}",
                    "source_contract_id": safe_id,
                    "category": cat,
                    "question": nl_question,
                    "is_unanswerable": True,
                    "gold_evidence": "",
                    "gold_answer_start": -1,
                    "gold_answers_count": 0,
                })

        dev_contracts.append({
            "source_contract_id": safe_id,
            "original_title": title,
            "filename": f"{safe_id}.md",
            "sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "char_length": len(context),
            "total_qas_in_source": len(qas),
            "valid_annotations_count": valid_annotations,
        })

    ans_count = sum(1 for q in dev_queries if not q["is_unanswerable"])
    unans_count = sum(1 for q in dev_queries if q["is_unanswerable"])

    manifest_data = {
        "dataset_name": "CUAD v1 DEV Tuning Set",
        "split": "DEV_TUNING",
        "total_contracts": len(dev_contracts),
        "total_queries": len(dev_queries),
        "total_answerable_queries": ans_count,
        "total_unanswerable_queries": unans_count,
        "contract_indices": dev_indices,
        "contracts": dev_contracts,
        "queries": dev_queries,
    }

    out_path = MANIFEST_DIR / "cuad_dev_manifest.json"
    out_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"[OK] Wrote DEV manifest to {out_path}")
    print(f"     Contracts: {len(dev_contracts)}")
    print(f"     Total Queries: {len(dev_queries)} (Answerable: {ans_count}, Unanswerable: {unans_count})")

if __name__ == "__main__":
    build_dev_split()
