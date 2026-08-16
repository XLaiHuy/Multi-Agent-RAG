# Reproducibility & Benchmark Execution Guide

This guide provides exact commands to run test suites, verify cache acceleration, and reproduce the Phase 4.2 frozen evaluation metrics.

---

## 1. Quality Gate & Unit Tests

Run all 54 unit, security, and benchmark integrity tests:
```bash
pytest tests/
```

Expected output:
```text
============================= 54 passed in ~16s =============================
```

---

## 2. Reproduce Frozen Holdout Evaluation

To execute the single-pass evaluation on `CUSTOM_CUAD_HOLDOUT_V2` ($N=294$) under strict child gold evidence mapping:
```bash
python evaluation/scripts/run_phase4_2.py
```

Generated result artifacts:
- `evaluation/results/phase4_2/final_holdout_strict_child_gold.json`
- `evaluation/results/phase4_2/gold_mapping_audit.json`
- `evaluation/results/phase4_2/online_latency_holdout.json`
- `evaluation/results/phase4_2/holdout_rank_trace_strict.jsonl`

---

## 3. Reproduce Cache Speedup Benchmark

To measure cold vs warm runtime and verify exact SHA-256 fingerprint matching:
```bash
python evaluation/scripts/benchmark_eval_cache.py
```

Expected output:
```text
  Cold Runtime: ~180 s
  Warm Runtime: ~1.54 s
  Speedup Ratio: ~116.8x
  Result Hash Match: YES (Exact Match)
```

Generated artifact:
- `evaluation/results/phase4_2/cache_speedup_verified.json`
