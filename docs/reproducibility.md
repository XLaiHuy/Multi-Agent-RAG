# Reproducibility Guide & Benchmark Execution

This guide provides instructions for reproducing all test suites, evaluation harnesses, and verifying benchmark artifacts in the repository.

---

## 1. Prerequisites & Environment Setup

Clone repository and configure Python environment:

```bash
# Clone repository
git clone https://github.com/XLaiHuy/Multi-Agent-RAG.git
cd Multi-Agent-RAG

# Create virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create local environment configuration:
```bash
cp .env.example .env
```

---

## 2. Running Automated Tests

Run the complete test suite (Unit tests, Security ACLs, Anti-IDOR, Token chunking, Deterministic agents):

```bash
python -m pytest tests/
```

*Expected Output*: `47 passed in ~15-25s`.

---

## 3. Reproducing Fast Evaluation Harness

The evaluation harness uses cryptographic parameter hashing (`evaluation/cache_manager.py`) to cache intermediate document chunks, dense embeddings, and BM25 tokenizations:

```bash
# Run Phase 4.1 DEV Evaluation (Warm cache: ~25.8s, Cold: ~40 min)
python evaluation/scripts/run_phase4_1.py
```

*Outputs generated in `evaluation/results/phase4_1/`:*
- `cache_speedup_apples_to_apples.json` (94.70x speedup verification)
- `true_doc_scoped_dev.json` (Global vs True Scoped metrics)
- `candidate_budget_dev.json` (Candidate budget sweep $k \in [10, 75]$)
- `reranker_ab_dev.json` (TinyBERT vs BGE-Reranker-Base A/B)
- `retrieval_latency_dev.json` (Live measured latency profiling)

---

## 4. Inspecting Frozen Held-Out Benchmark

The canonical evaluation on `CUSTOM_CUAD_HOLDOUT_V2` ($N=293$) is frozen under configuration `v4.1.0` (`evaluation/configs/retrieval_final_config_v4_1.json`):

```bash
python -c "import json; res=json.load(open('evaluation/results/phase4_1/final_holdout_doc_scoped.json')); print('Hit@10:', res['post_rerank_metrics']['HitRate@10'], 'MRR:', res['post_rerank_metrics']['MRR'])"
```

*Canonical Held-Out Result:*
- **Hit@5**: 82.94%
- **Hit@10**: 94.54%
- **MRR**: 0.6418
- **Latency P50**: 68.89 ms (CPU)
