"""
Script to prepare:
1. Full UIT-ViQuAD 2.0 Validation Split (N=3,814 samples)
2. Stanford SQuAD 2.0 Benchmark (N=2,000 samples)
"""
import json
import sys
from pathlib import Path
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("data/evaluation")
VIQUAD_DIR = DATA_DIR / "viquad_full"
SQUAD_DIR = DATA_DIR / "squad_2k"

VIQUAD_DIR.mkdir(parents=True, exist_ok=True)
SQUAD_DIR.mkdir(parents=True, exist_ok=True)


def prepare_full_viquad():
    print("[1/2] Loading FULL UIT-ViQuAD 2.0 validation split...")
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="validation")
    total = len(ds)
    print(f"  -> Total samples loaded: {total}")

    context_to_id = {}
    corpus = []
    eval_items = []

    for i, item in enumerate(ds):
        ctx = item["context"].strip()
        if ctx not in context_to_id:
            cid = f"viquad_ctx_{len(context_to_id):05d}"
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
            "id": f"viquad_{i+1:05d}",
            "question": item["question"].strip(),
            "title": item.get("title", ""),
            "expected_chunk_ids": [cid] if not is_impossible else [],
            "expected_answer": expected_ans,
            "is_unanswerable": is_impossible,
        })

    (VIQUAD_DIR / "corpus_full.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    (VIQUAD_DIR / "viquad_full_eval.json").write_text(json.dumps(eval_items, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Saved {len(corpus)} unique context passages and {len(eval_items)} QA pairs.")


def prepare_squad_benchmark(target_count: int = 2000):
    print(f"[2/2] Loading Stanford SQuAD 2.0 validation split (N={target_count})...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    total_avail = len(ds)
    print(f"  -> Available SQuAD v2 samples: {total_avail}. Selecting top {target_count}...")

    subset = ds.select(range(min(target_count, total_avail)))

    context_to_id = {}
    corpus = []
    eval_items = []

    for i, item in enumerate(subset):
        ctx = item["context"].strip()
        if ctx not in context_to_id:
            cid = f"squad_ctx_{len(context_to_id):05d}"
            context_to_id[ctx] = cid
            corpus.append({
                "chunk_id": cid,
                "title": item.get("title", ""),
                "text": ctx,
            })
        else:
            cid = context_to_id[ctx]

        answers = item.get("answers", {}).get("text", [])
        is_impossible = len(answers) == 0
        expected_ans = answers[0] if answers else "Unanswerable"

        eval_items.append({
            "id": f"squad_{i+1:05d}",
            "question": item["question"].strip(),
            "title": item.get("title", ""),
            "expected_chunk_ids": [cid] if not is_impossible else [],
            "expected_answer": expected_ans,
            "is_unanswerable": is_impossible,
        })

    (SQUAD_DIR / "corpus_2k.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    (SQUAD_DIR / "squad_2k_eval.json").write_text(json.dumps(eval_items, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Saved {len(corpus)} SQuAD passages and {len(eval_items)} SQuAD QA pairs.")


if __name__ == "__main__":
    prepare_full_viquad()
    prepare_squad_benchmark(2000)
    print("[Done] Prepared both datasets successfully!")
