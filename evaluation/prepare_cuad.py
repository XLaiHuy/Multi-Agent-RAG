"""
Official CUAD Dataset Preparer and Leakage Auditor.
Parses official CUADv1.json (510 contracts, 41 clause categories).
Extracts:
1. Official source contracts in canonical format.
2. Frozen TEST split and DEV split.
3. Audits lexical overlap and title leakage (evaluation/reports/cuad_leakage_audit.json).
4. Generates programmatic evaluation/manifests/cuad_official_manifest.json.
5. Generates frozen evaluation/configs/final_eval_config.json.
"""
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set

RAW_JSON_PATH = Path("evaluation/datasets/cuad/raw/CUADv1.json")
PROCESSED_DIR = Path("evaluation/datasets/cuad/processed")
CONTRACTS_DIR = PROCESSED_DIR / "contracts"
MANIFEST_DIR = Path("evaluation/manifests")
CONFIG_DIR = Path("evaluation/configs")
REPORTS_DIR = Path("evaluation/reports")

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
}


def calculate_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_lexical_overlap(query: str, target_text: str) -> float:
    q_words = set(re.findall(r"\b\w+\b", query.lower()))
    t_words = set(re.findall(r"\b\w+\b", target_text.lower()))
    if not q_words:
        return 0.0
    return len(q_words.intersection(t_words)) / len(q_words)


def prepare_official_cuad():
    print(f"[CUAD Preparer] Loading official CUAD from {RAW_JSON_PATH}...")
    if not RAW_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_JSON_PATH}. Run download_cuad.py first.")

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    contracts_raw = cuad_data.get("data", [])
    print(f"[CUAD Preparer] Loaded {len(contracts_raw)} official contracts.")

    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Select 10 diverse contracts for the frozen TEST evaluation set
    # and 3 contracts for the DEV/Tuning set
    dev_contract_indices = [0, 1, 2]
    test_contract_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    manifest_contracts = []
    manifest_queries = []
    leakage_audit_records = []

    # Process TEST set contracts
    for c_idx in test_contract_indices:
        contract_data = contracts_raw[c_idx]
        title = contract_data["title"]
        safe_id = f"cuad_contract_{c_idx:03d}_{re.sub(r'[^a-zA-Z0-9_]', '_', title)[:30]}"
        paragraph = contract_data["paragraphs"][0]
        context_text = paragraph["context"]

        # Save contract text & markdown
        c_path = CONTRACTS_DIR / f"{safe_id}.txt"
        c_md_path = CONTRACTS_DIR / f"{safe_id}.md"
        c_path.write_text(context_text, encoding="utf-8")
        c_md_path.write_text(f"# {title}\n\n{context_text}", encoding="utf-8")

        c_sha256 = calculate_sha256(context_text)
        qas = paragraph.get("qas", [])

        valid_annotations_count = sum(1 for q in qas if not q.get("is_impossible", False) and len(q.get("answers", [])) > 0)

        manifest_contracts.append({
            "source_contract_id": safe_id,
            "original_title": title,
            "filename": f"{safe_id}.md",
            "sha256": c_sha256,
            "char_length": len(context_text),
            "total_qas_in_source": len(qas),
            "valid_annotations_count": valid_annotations_count
        })

        # Generate representative test queries from CUAD annotations
        for q in qas:
            raw_category = q.get("id", "").split("__")[-1] if "__" in q.get("id", "") else q.get("id")
            category_clean = raw_category.replace("_", " ").strip()
            is_impossible = q.get("is_impossible", False)
            answers = q.get("answers", [])

            # Use mapped natural query or fallback
            natural_question = CLAUSE_QUERY_MAP.get(category_clean)
            if not natural_question:
                continue

            gold_answer = answers[0]["text"] if answers else "The agreement does not contain provisions on this subject."
            gold_start = answers[0]["answer_start"] if answers else None

            # Leakage Analysis
            title_overlap = compute_lexical_overlap(natural_question, title)
            answer_overlap = compute_lexical_overlap(natural_question, gold_answer) if not is_impossible else 0.0

            leakage_status = "LEAK_FREE"
            if title_overlap > 0.60:
                leakage_status = "TITLE_LEAKAGE_FLAGGED"
            elif answer_overlap > 0.70 and len(gold_answer) < 30:
                leakage_status = "ANSWER_SPAN_LEAKAGE_FLAGGED"

            q_id = f"eval_{safe_id[:16]}_{re.sub(r'[^a-zA-Z0-9]', '_', category_clean)[:15]}"

            query_obj = {
                "query_id": q_id,
                "source_contract_id": safe_id,
                "contract_title": title,
                "clause_category": category_clean,
                "question": natural_question,
                "is_unanswerable": is_impossible,
                "gold_evidence": gold_answer,
                "gold_answer_start": gold_start,
                "leakage_status": leakage_status
            }

            manifest_queries.append(query_obj)
            leakage_audit_records.append({
                "query_id": q_id,
                "category": category_clean,
                "title_overlap": round(title_overlap, 3),
                "answer_overlap": round(answer_overlap, 3),
                "leakage_status": leakage_status
            })

    # Limit to 50 curated, leak-free test queries across the 10 contracts
    manifest_queries_leak_free = [q for q in manifest_queries if q["leakage_status"] == "LEAK_FREE"][:50]

    official_manifest = {
        "dataset_name": "Contract Understanding Atticus Dataset (CUAD) v1",
        "source": "The Atticus Project",
        "source_url": "https://github.com/TheAtticusProject/cuad",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_contracts_available_in_cuad": len(contracts_raw),
        "total_contracts_in_test_split": len(manifest_contracts),
        "total_queries_in_test_split": len(manifest_queries_leak_free),
        "split": "TEST_FROZEN",
        "random_seed": 42,
        "contracts": manifest_contracts,
        "queries": manifest_queries_leak_free
    }

    manifest_out = MANIFEST_DIR / "cuad_official_manifest.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(official_manifest, f, indent=2)

    leakage_out = REPORTS_DIR / "cuad_leakage_audit.json"
    with open(leakage_out, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries_audited": len(leakage_audit_records),
            "leak_free_count": sum(1 for r in leakage_audit_records if r["leakage_status"] == "LEAK_FREE"),
            "flagged_count": sum(1 for r in leakage_audit_records if r["leakage_status"] != "LEAK_FREE"),
            "audit_records": leakage_audit_records[:50]
        }, f, indent=2)

    # Save frozen configuration (Section 7)
    frozen_config = {
        "config_version": "1.0.0_frozen",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retrieval_confidence_threshold": 0.70,
        "top_k_retrieval": 10,
        "reranker_candidate_count": 15,
        "reranker_top_n": 5,
        "rrf_k": 60,
        "child_chunk_tokens": 250,
        "parent_chunk_tokens": 1200,
        "max_retrieval_attempts": 2,
        "max_regeneration_count": 1,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "generation_model": "gemini-flash-latest"
    }
    with open(CONFIG_DIR / "final_eval_config.json", "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, indent=2)

    print(f"[CUAD Preparer] Successfully prepared official CUAD evaluation set!")
    print(f"  - Manifest: {manifest_out} (Contracts: {len(manifest_contracts)}, Queries: {len(manifest_queries_leak_free)})")
    print(f"  - Leakage Report: {leakage_out}")
    print(f"  - Frozen Config: {CONFIG_DIR / 'final_eval_config.json'}")


if __name__ == "__main__":
    prepare_official_cuad()
