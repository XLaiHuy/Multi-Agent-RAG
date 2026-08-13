"""
Script to extract and prepare 1,000 benchmark QA pairs and unique context corpus
from UIT-ViQuAD 2.0 (University of Information Technology, VNU-HCM).
"""
import json
import sys
from pathlib import Path
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("data/evaluation/viquad_1k")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = OUT_DIR / "corpus_1k.json"
DATASET_PATH = OUT_DIR / "viquad_1k_eval.json"


def prepare_viquad_1k():
    print("Loading UIT-ViQuAD2.0 validation split...")
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="validation")
    print(f"Total available validation samples: {len(ds)}")

    # Take first 1,000 samples
    target_count = 1000
    subset = ds.select(range(target_count))

    # Map unique contexts to chunk_ids
    context_to_id = {}
    corpus = []
    eval_items = []

    for i, item in enumerate(subset):
        ctx = item["context"].strip()
        if ctx not in context_to_id:
            cid = f"viquad_ctx_{len(context_to_id):04d}"
            context_to_id[ctx] = cid
            corpus.append({
                "chunk_id": cid,
                "title": item.get("title", ""),
                "text": ctx,
            })
        else:
            cid = context_to_id[ctx]

        is_impossible = item.get("is_impossible", False)
        answers = item.get("answers", {}).get("text", [])
        expected_ans = answers[0] if answers else ("Unanswerable" if is_impossible else "")

        eval_items.append({
            "id": f"viquad_{i+1:04d}",
            "question": item["question"].strip(),
            "title": item.get("title", ""),
            "expected_chunk_ids": [cid] if not is_impossible else [],
            "expected_answer": expected_ans,
            "is_unanswerable": is_impossible,
        })

    # Save to files
    CORPUS_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    DATASET_PATH.write_text(json.dumps(eval_items, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] Prepared {len(eval_items)} eval items from {len(corpus)} unique context passages.")
    print(f"  - Answerable items: {sum(1 for x in eval_items if not x['is_unanswerable'])}")
    print(f"  - Unanswerable items: {sum(1 for x in eval_items if x['is_unanswerable'])}")
    print(f"  - Corpus saved to: {CORPUS_PATH}")
    print(f"  - Eval dataset saved to: {DATASET_PATH}")


if __name__ == "__main__":
    prepare_viquad_1k()
