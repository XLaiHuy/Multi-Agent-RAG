# Agent Ablation & Routing Report (Historical Reference)

> **SCIENTIFIC INTEGRITY NOTICE (Phase 3.5 Repair):**  
> The metrics in this document represent a **preliminary simulation pass**.  
> The evaluation script used heuristic score thresholding and contained benchmark label access.  
> These metrics are classified as **`SIMULATION_ONLY`** and are **`NOT_CV_SAFE`** until live LLM API benchmarks are executed with strict label isolation (`execute_query_without_gold`).

---

## 1. Simulation Findings (Historical Log)

- **Simulated Refusal Rate:** 31/31 unanswerable queries refused via deterministic confidence thresholds.
- **Simulated Routing Distribution:** Fast path (Level 1) vs Escalated verification (Level 2).
- **Framework Status:** Code refactored with clean `execute_query_without_gold` and `score_execution_against_gold` interfaces in `evaluation/scripts/run_agent_ablation.py`.
