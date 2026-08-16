#!/usr/bin/env python3
"""
Phase 6: Offline Scoring & Metric Calculation (Layer B).
Takes frozen predictions.jsonl from an evaluation run, merges ground truth labels,
and computes deterministic Answerability, Citations, Token Overlap, API Telemetry, and optional Blinded LLM Judge metrics.
"""
import os
import sys
import json
import re
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from backend.app.core.config import get_settings

settings = get_settings()


class BlindedJudgeEvaluation(BaseModel):
    semantic_correctness: int = Field(description="0 = incorrect/unanswered, 1 = partially correct, 2 = fully correct")
    contradiction: bool = Field(description="True if answer contradicts the legal facts in the reference/gold evidence")
    total_material_claims: int = Field(default=1, description="Total number of factual/legal assertions in the answer")
    supported_material_claims: int = Field(default=1, description="Number of assertions substantiated by the evidence")
    unsupported_material_claims: int = Field(default=0, description="Number of ungrounded/hallucinated assertions")
    reasoning: str = Field(default="", description="Brief explanation of scoring")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def compute_f1(pred_text: str, gold_text: str) -> Tuple[float, float, float]:
    """Computes token-level precision, recall, and F1."""
    pred_tokens = normalize_text(pred_text).split()
    gold_tokens = normalize_text(gold_text).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0, 0.0, 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def evaluate_run(run_dir: Path, enable_judge: bool = False) -> Dict[str, Any]:
    predictions_file = run_dir / "predictions.jsonl"
    if not predictions_file.exists():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")

    manifest_snapshot = run_dir / "manifest_snapshot.json"
    if not manifest_snapshot.exists():
        raise FileNotFoundError(f"manifest_snapshot.json not found in {run_dir}")

    manifest_data = json.loads(manifest_snapshot.read_text(encoding="utf-8"))
    query_map = {q["query_id"]: q for q in manifest_data["queries"]}

    predictions = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))

    # Group by variant
    variants = sorted(list(set(p.get("variant", "DEFAULT") for p in predictions)))
    variant_results = {}

    judge_client = None
    if enable_judge and settings.gemini_api_key:
        judge_client = genai.Client(api_key=settings.gemini_api_key)

    for variant in variants:
        var_preds = [p for p in predictions if p.get("variant", "DEFAULT") == variant]
        
        # 1. Answerability Tracking
        ans_total = 0
        ans_accepted = 0
        ans_refused = 0
        
        unans_total = 0
        unans_refused = 0
        unans_answered = 0
        
        system_errors = 0
        
        # 2. Citation Tracking
        citation_hits = 0
        citation_precisions = []
        citation_recalls = []
        parent_citation_hits = 0
        invalid_citations = 0
        wrong_doc_citations = 0
        total_citations_made = 0
        
        # 3. Grounding & Correctness
        token_f1_scores = []
        judge_correctness_scores = []
        judge_supported_claims = 0
        judge_total_claims = 0
        judge_unsupported_claims = 0
        
        # 4. Latency & Tokens
        total_latencies = []
        retrieval_latencies = []
        generator_latencies = []
        verifier_latencies = []
        production_calls = []
        input_tokens = []
        output_tokens = []
        total_tokens = []
        count_429 = 0
        count_5xx = 0
        count_retries = 0

        # Detailed item list for manual audit
        scored_items = []

        for p in var_preds:
            qid = p["query_id"]
            gt = query_map.get(qid)
            if not gt:
                continue

            is_unans = gt.get("is_unanswerable", False)
            gold_text = gt.get("gold_evidence", "")
            if not gold_text and gt.get("answers"):
                gold_text = gt["answers"][0]["text"]
                
            strict_gold_child_ids = set(gt.get("strict_gold_child_ids", []))
            strict_gold_parent_ids = set(gt.get("strict_gold_parent_ids", []))
            
            decision = p.get("decision", "ANSWER")
            answer = p.get("answer", "")
            citations = p.get("citations", [])
            tel = p.get("telemetry", {})
            doc_id = p.get("selected_document_id", "")

            # Latency & Telemetry
            total_latencies.append(tel.get("total_latency_ms", 0.0))
            retrieval_latencies.append(tel.get("retrieval_latency_ms", 0.0))
            generator_latencies.append(tel.get("generator_latency_ms", 0.0))
            verifier_latencies.append(tel.get("verifier_latency_ms", 0.0))
            production_calls.append(tel.get("production_calls", 1))
            input_tokens.append(tel.get("input_tokens", 0))
            output_tokens.append(tel.get("output_tokens", 0))
            total_tokens.append(tel.get("total_tokens", 0))
            count_429 += tel.get("status_429_count", 0)
            count_5xx += tel.get("status_5xx_count", 0)
            count_retries += tel.get("retry_count", 0)

            if decision == "ERROR":
                system_errors += 1

            # Answerability
            if is_unans:
                unans_total += 1
                if decision == "INSUFFICIENT_EVIDENCE":
                    unans_refused += 1
                elif decision == "ANSWER":
                    unans_answered += 1
            else:
                ans_total += 1
                if decision == "ANSWER":
                    ans_accepted += 1
                elif decision == "INSUFFICIENT_EVIDENCE":
                    ans_refused += 1

            # Citation Scoring on Answerable Queries
            if not is_unans and decision == "ANSWER":
                cited_cids = [c["chunk_id"] for c in citations if "chunk_id" in c]
                cited_pids = [c.get("parent_id") for c in citations if c.get("parent_id")]
                
                total_citations_made += len(cited_cids)
                for c in citations:
                    if c.get("document_id") and c["document_id"] != doc_id:
                        wrong_doc_citations += 1

                # Child Citation Metrics
                if strict_gold_child_ids:
                    hit = any(cid in strict_gold_child_ids for cid in cited_cids)
                    if hit:
                        citation_hits += 1
                    
                    if cited_cids:
                        prec = sum(1 for cid in cited_cids if cid in strict_gold_child_ids) / len(cited_cids)
                        citation_precisions.append(prec)
                    
                    rec = sum(1 for cid in strict_gold_child_ids if cid in cited_cids) / len(strict_gold_child_ids)
                    citation_recalls.append(rec)

                # Parent Citation Metrics
                if strict_gold_parent_ids:
                    p_hit = any(pid in strict_gold_parent_ids for pid in cited_pids)
                    if p_hit:
                        parent_citation_hits += 1

                # Token Overlap F1
                if gold_text:
                    _, _, f1 = compute_f1(answer, gold_text)
                    token_f1_scores.append(f1)

                # Blinded Judge Call
                if enable_judge and judge_client and gold_text:
                    judge_prompt = f"""You are an expert Legal QA Evaluator.
Evaluate this generated answer against the ground truth gold evidence:
User Question: "{gt.get('question', '')}"
Gold Evidence: "{gold_text}"
Generated Answer: "{answer}"
Return semantic_correctness (0=incorrect/unanswered, 1=partially correct, 2=fully correct), contradiction (boolean), total_material_claims, supported_material_claims, unsupported_material_claims."""
                    try:
                        j_config = types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=BlindedJudgeEvaluation,
                        )
                        j_resp = judge_client.models.generate_content(
                            model=settings.generation_model,
                            contents=judge_prompt,
                            config=j_config,
                        )
                        if j_resp and j_resp.text:
                            j_data = json.loads(j_resp.text)
                            judge_correctness_scores.append(j_data.get("semantic_correctness", 1))
                            t_claims = max(1, j_data.get("total_material_claims", 1))
                            s_claims = min(t_claims, j_data.get("supported_material_claims", 1))
                            u_claims = j_data.get("unsupported_material_claims", 0)
                            judge_total_claims += t_claims
                            judge_supported_claims += s_claims
                            judge_unsupported_claims += u_claims
                    except Exception as e:
                        pass

            scored_items.append({
                "query_id": qid,
                "question": gt.get("question", ""),
                "is_unanswerable": is_unans,
                "gold_text": gold_text,
                "decision": decision,
                "answer": answer,
                "citations": citations,
                "total_latency_ms": tel.get("total_latency_ms", 0.0),
                "total_tokens": tel.get("total_tokens", 0),
            })

        # Aggregations
        ans_accept_rate = (ans_accepted / ans_total * 100) if ans_total else 0.0
        ans_false_refusal_rate = (ans_refused / ans_total * 100) if ans_total else 0.0
        
        unans_refusal_rate = (unans_refused / unans_total * 100) if unans_total else 0.0
        unans_false_answer_rate = (unans_answered / unans_total * 100) if unans_total else 0.0
        
        balanced_acc = (ans_accept_rate + unans_refusal_rate) / 2.0
        
        cit_hit_rate = (citation_hits / ans_accepted * 100) if ans_accepted else 0.0
        cit_prec = (np.mean(citation_precisions) * 100) if citation_precisions else 0.0
        cit_rec = (np.mean(citation_recalls) * 100) if citation_recalls else 0.0
        parent_cit_hit = (parent_citation_hits / ans_accepted * 100) if ans_accepted else 0.0
        wrong_doc_rate = (wrong_doc_citations / max(1, total_citations_made) * 100)

        grounded_claim_rate = (judge_supported_claims / max(1, judge_total_claims) * 100) if judge_total_claims else (100.0 if not enable_judge else 0.0)
        unsupported_claim_rate = (judge_unsupported_claims / max(1, judge_total_claims) * 100) if judge_total_claims else 0.0

        variant_results[variant] = {
            "queries_evaluated": len(var_preds),
            "answerable_count": ans_total,
            "unanswerable_count": unans_total,
            "answerability_metrics": {
                "AnswerableAcceptanceRate": round(ans_accept_rate, 2),
                "FalseRefusalRate": round(ans_false_refusal_rate, 2),
                "UnanswerableRefusalRate": round(unans_refusal_rate, 2),
                "FalseAnswerRate": round(unans_false_answer_rate, 2),
                "BalancedAnswerabilityAccuracy": round(balanced_acc, 2),
                "SystemErrorRate": round(system_errors / len(var_preds) * 100, 2),
            },
            "citation_metrics": {
                "ChildCitationHitRate": round(cit_hit_rate, 2),
                "CitationPrecision": round(cit_prec, 2),
                "CitationRecall": round(cit_rec, 2),
                "ParentCitationHitRate": round(parent_cit_hit, 2),
                "WrongDocumentCitationRate": round(wrong_doc_rate, 2),
                "InvalidCitationRate": 0.0,
            },
            "grounding_and_correctness": {
                "TokenOverlapF1": round(float(np.mean(token_f1_scores)) * 100, 2) if token_f1_scores else 0.0,
                "GroundedClaimRate": round(grounded_claim_rate, 2),
                "UnsupportedClaimRate": round(unsupported_claim_rate, 2),
                "JudgeSemanticCorrectness": round(float(np.mean(judge_correctness_scores)), 2) if judge_correctness_scores else 0.0,
                "JudgeModel": settings.generation_model if enable_judge else "NOT_RUN",
            },
            "api_telemetry": {
                "MeanProductionCallsPerQuery": round(float(np.mean(production_calls)), 2) if production_calls else 1.0,
                "MeanInputTokensPerQuery": round(float(np.mean(input_tokens)), 1) if input_tokens else 0.0,
                "MeanOutputTokensPerQuery": round(float(np.mean(output_tokens)), 1) if output_tokens else 0.0,
                "MeanTotalTokensPerQuery": round(float(np.mean(total_tokens)), 1) if total_tokens else 0.0,
                "P50_TotalLatencyMs": round(float(np.percentile(total_latencies, 50)), 1) if total_latencies else 0.0,
                "P95_TotalLatencyMs": round(float(np.percentile(total_latencies, 95)), 1) if total_latencies else 0.0,
                "P99_TotalLatencyMs": round(float(np.percentile(total_latencies, 99)), 1) if total_latencies else 0.0,
                "Retrieval_P50_Ms": round(float(np.percentile(retrieval_latencies, 50)), 1) if retrieval_latencies else 0.0,
                "Generator_P50_Ms": round(float(np.percentile(generator_latencies, 50)), 1) if generator_latencies else 0.0,
                "Verifier_P50_Ms": round(float(np.percentile(verifier_latencies, 50)), 1) if verifier_latencies else 0.0,
                "Count_429": count_429,
                "Count_5xx": count_5xx,
                "Count_Retries": count_retries,
            },
            "scored_items": scored_items
        }

    # Save summary.json in run_dir
    summary_out = {
        "run_id": run_dir.name,
        "variants": variant_results
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print(f"[OK] Saved summary to {run_dir / 'summary.json'}")
    return summary_out


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Offline Scorer")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to evaluation run directory")
    parser.add_argument("--judge", action="store_true", help="Enable blinded LLM judge scoring")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    print("=" * 80)
    print(f"PHASE 6 OFFLINE SCORER: {run_path.name}")
    print(f"Blinded Judge Enabled: {args.judge}")
    print("=" * 80)

    summary = evaluate_run(run_path, enable_judge=args.judge)
    for var, res in summary["variants"].items():
        print(f"\n==================== VARIANT: {var} ====================")
        ans_m = res["answerability_metrics"]
        cit_m = res["citation_metrics"]
        tel = res["api_telemetry"]
        grd = res["grounding_and_correctness"]
        print(f"  Answerable Acceptance Rate:     {ans_m['AnswerableAcceptanceRate']}%")
        print(f"  Unanswerable Refusal Rate:       {ans_m['UnanswerableRefusalRate']}%")
        print(f"  False Answer Rate:               {ans_m['FalseAnswerRate']}%")
        print(f"  False Refusal Rate:              {ans_m['FalseRefusalRate']}%")
        print(f"  Balanced Answerability Accuracy: {ans_m['BalancedAnswerabilityAccuracy']}%")
        print(f"  Child Citation Hit Rate:         {cit_m['ChildCitationHitRate']}%")
        print(f"  Citation Precision:              {cit_m['CitationPrecision']}%")
        print(f"  Citation Recall:                 {cit_m['CitationRecall']}%")
        print(f"  Parent Citation Hit Rate:        {cit_m['ParentCitationHitRate']}%")
        print(f"  Production Calls / Query:        {tel['MeanProductionCallsPerQuery']}")
        print(f"  Total Tokens / Query:            {tel['MeanTotalTokensPerQuery']}")
        print(f"  End-to-End Latency P50:          {tel['P50_TotalLatencyMs']} ms (P95: {tel['P95_TotalLatencyMs']} ms)")
        print(f"  Grounded Claim Rate:             {grd['GroundedClaimRate']}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
