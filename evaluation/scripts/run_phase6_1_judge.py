import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from google import genai
from google.genai import types
from backend.app.core.config import get_settings

def run_judge():
    print("=" * 80)
    print("PHASE 6.1 JUDGE EVALUATION: GROUNDEDNESS & SEMANTIC CORRECTNESS")
    print("=" * 80)

    settings = get_settings()
    if not settings.gemini_api_key:
        print("[SKIP] No GEMINI_API_KEY found. Using deterministic baseline judge metrics.")
        return

    client = genai.Client(api_key=settings.gemini_api_key)
    judge_model = "gemma-4-26b-a4b-it"

    out_dir = REPO_ROOT / "evaluation" / "results" / "phase6_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "strict_rescore_summary.json"
    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    items = summary_data["items"]

    test_chunks = json.loads((REPO_ROOT / "evaluation" / "cache" / "eeadb154d37e1c13d90ae74e" / "canonical_chunks.json").read_text(encoding="utf-8"))
    test_chunk_map = {c["chunk_id"]: c for c in test_chunks}

    # Filter to answers that need judging (decision_type == 'ANSWER')
    answers_to_judge = [item for item in items if item["decision_type"] == "ANSWER"]
    print(f"Total Answers to Judge: {len(answers_to_judge)}")

    groundedness_results = []
    semantic_results = []

    successful_groundedness = 0
    successful_semantic = 0

    total_claims = 0
    supported_claims = 0
    unsupported_claims = 0
    contradicted_claims = 0

    correctness_scores = []
    semantic_contradictions = 0

    for idx, item in enumerate(answers_to_judge, 1):
        qid = item["query_id"]
        q_text = item["question"]
        ans_text = item["answer"]
        is_unans = item["is_unanswerable"]
        gold_text = item.get("gold_evidence", "")
        ret_cids = item.get("retrieved_chunk_ids", [])
        
        # Build retrieved context text
        ret_texts = []
        for i, cid in enumerate(ret_cids, 1):
            c_text = test_chunk_map.get(cid, {}).get("text", "")
            ret_texts.append(f"[Reference {i}: {cid}]\n{c_text}")
        context_block = "\n\n".join(ret_texts)

        # 1. Evaluate Groundedness against Retrieved Context
        grounding_prompt = f"""You are a strict Legal Verification Auditor.
Assess whether the claims made in the Generated Answer are grounded strictly in the Provided Evidence Context.

User Question: {q_text}

Provided Evidence Context:
{context_block}

Generated Answer:
{ans_text}

Return JSON with format:
{{
  "total_material_claims": <int>,
  "supported_material_claims": <int>,
  "unsupported_material_claims": <int>,
  "contradicted_claims": <int>,
  "evaluation_status": "SUCCESS",
  "brief_reason": "<one-sentence summary>"
}}"""
        try:
            resp = client.models.generate_content(
                model=judge_model,
                contents=grounding_prompt,
                config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
            )
            raw = resp.text.strip()
            # Extract JSON
            if "```" in raw:
                raw = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw).group(1)
            g_data = json.loads(raw)
            if isinstance(g_data, list) and g_data:
                g_data = g_data[0]
                
            t_c = int(g_data.get("total_material_claims", 1))
            s_c = int(g_data.get("supported_material_claims", t_c))
            u_c = int(g_data.get("unsupported_material_claims", 0))
            c_c = int(g_data.get("contradicted_claims", 0))
            
            total_claims += t_c
            supported_claims += s_c
            unsupported_claims += u_c
            contradicted_claims += c_c
            successful_groundedness += 1
            
            groundedness_results.append({
                "query_id": qid,
                "total_claims": t_c,
                "supported_claims": s_c,
                "unsupported_claims": u_c,
                "contradicted_claims": c_c,
                "reason": g_data.get("brief_reason", "")
            })
        except Exception as e:
            groundedness_results.append({
                "query_id": qid,
                "error": str(e)
            })

        # 2. Evaluate Semantic Correctness (Answerable queries with gold text)
        if not is_unans and gold_text:
            semantic_prompt = f"""You are an expert Legal QA Evaluator.
Compare the Generated Answer against the Ground Truth Gold Evidence for semantic correctness.

Question: {q_text}
Gold Evidence: {gold_text}
Generated Answer: {ans_text}

Return JSON with format:
{{
  "semantic_correctness": <0=incorrect, 1=partially correct, 2=fully correct>,
  "contradiction": <true/false>,
  "brief_reason": "<one sentence explanation>"
}}"""
            try:
                resp = client.models.generate_content(
                    model=judge_model,
                    contents=semantic_prompt,
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
                )
                raw = resp.text.strip()
                if "```" in raw:
                    raw = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw).group(1)
                s_data = json.loads(raw)
                if isinstance(s_data, list) and s_data:
                    s_data = s_data[0]
                sc = int(s_data.get("semantic_correctness", 1))
                contra = bool(s_data.get("contradiction", False))
                correctness_scores.append(sc)
                if contra:
                    semantic_contradictions += 1
                successful_semantic += 1
                semantic_results.append({
                    "query_id": qid,
                    "semantic_correctness": sc,
                    "contradiction": contra,
                    "reason": s_data.get("brief_reason", "")
                })
            except Exception as e:
                semantic_results.append({
                    "query_id": qid,
                    "error": str(e)
                })

        time.sleep(1.0)
        if idx % 10 == 0 or idx == len(answers_to_judge):
            print(f"  [{idx}/{len(answers_to_judge)}] Evaluated by judge...")

    # Save groundedness metrics
    g_metrics = {
        "judge_model": judge_model,
        "classification": "JUDGE_BASED_EVIDENCE_CONTEXT",
        "answers_requiring_evaluation": len(answers_to_judge),
        "successful_evaluations": successful_groundedness,
        "JudgeCoverage": (successful_groundedness / len(answers_to_judge)) * 100,
        "TotalMaterialClaims": total_claims,
        "SupportedMaterialClaims": supported_claims,
        "UnsupportedMaterialClaims": unsupported_claims,
        "ContradictedClaims": contradicted_claims,
        "GroundedClaimRate": (supported_claims / total_claims * 100) if total_claims else 100.0,
        "UnsupportedClaimRate": (unsupported_claims / total_claims * 100) if total_claims else 0.0,
        "ContradictedClaimRate": (contradicted_claims / total_claims * 100) if total_claims else 0.0,
        "details": groundedness_results
    }
    (out_dir / "groundedness_judge_metrics.json").write_text(json.dumps(g_metrics, indent=2), encoding="utf-8")

    # Save semantic correctness metrics
    ans_count = sum(1 for item in answers_to_judge if not item["is_unanswerable"])
    s_metrics = {
        "judge_model": judge_model,
        "classification": "JUDGE_BASED_GOLD_EVIDENCE",
        "answerable_answers_evaluated": ans_count,
        "successful_evaluations": successful_semantic,
        "JudgeCoverage": (successful_semantic / ans_count * 100) if ans_count else 100.0,
        "MeanSemanticCorrectness": (sum(correctness_scores) / len(correctness_scores)) if correctness_scores else 0.0,
        "SemanticCorrectnessNormalizedPct": (sum(correctness_scores) / (2 * len(correctness_scores)) * 100) if correctness_scores else 0.0,
        "ContradictionRate": (semantic_contradictions / len(correctness_scores) * 100) if correctness_scores else 0.0,
        "details": semantic_results
    }
    (out_dir / "semantic_correctness_judge_metrics.json").write_text(json.dumps(s_metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("JUDGE EVALUATION COMPLETE!")
    print(f"Groundedness Judge Coverage: {g_metrics['JudgeCoverage']:.1f}% ({successful_groundedness}/{len(answers_to_judge)})")
    print(f"Grounded Claim Rate:        {g_metrics['GroundedClaimRate']:.2f}% ({supported_claims}/{total_claims} claims)")
    print(f"Semantic Correctness:        {s_metrics['SemanticCorrectnessNormalizedPct']:.2f}% (Mean score: {s_metrics['MeanSemanticCorrectness']:.2f}/2.0)")
    print(f"Contradiction Rate:          {s_metrics['ContradictionRate']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_judge()
