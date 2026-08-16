#!/usr/bin/env python3
"""
Phase 6: Canonical Real-API RAG Evaluation Runner.
Supports:
  --mode smoke          (N=20: 10 answerable, 10 unanswerable on DEV subset)
  --mode dev-ablation   (N=80: 40 answerable, 40 unanswerable across 3 variants: BASE_RAG, RAG_PLUS_VERIFIER, FULL_BOUNDED_MULTI_AGENT)
  --mode final          (N=200: 100 answerable, 100 unanswerable on CUSTOM_CUAD_HOLDOUT_V2 with winning system)
"""
import os
import sys
import time
import json
import random
import re
import argparse
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
torch.set_num_threads(4)

from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.app.core.config import get_settings
from backend.app.providers.reranker import LocalCrossEncoderReranker
from backend.app.providers.embeddings import LocalEmbeddingProvider
from backend.app.ingestion.parsers import MasterDocumentParser
from backend.app.ingestion.chunker import StructureAwareParentChildChunker
from backend.app.retrieval.bm25 import tokenize_for_bm25
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.agents.planner import RetrievalPlan, RetrievalPlannerAgent
from backend.app.agents.critic import EvidenceCriticEvaluation, EvidenceCriticAgent
from backend.app.agents.verifier import AnswerVerificationResult, AnswerVerifierAgent
from evaluation.cache_manager import EvaluationCache
from evaluation.config_loader import get_retrieval_config

settings = get_settings()
cfg = get_retrieval_config()
CONTRACTS_DIR = REPO_ROOT / "evaluation" / "datasets" / "cuad" / "processed" / "contracts"
RUNS_DIR = REPO_ROOT / "evaluation" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


class RealAPIGatewayClient:
    """Direct instrumented client wrapping google.genai with exact token and latency telemetry."""

    def __init__(self, api_key: str, default_model: str = "gemini-flash-latest"):
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.default_model = default_model

    def call_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = 1024,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Issues real text generation call with usage and latency tracking."""
        model_name = model or self.default_model
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
        )
        
        t0 = time.perf_counter()
        retries_used = 0
        status_429 = 0
        status_5xx = 0
        
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                
                text_out = resp.text if resp and resp.text else ""
                usage = getattr(resp, "usage_metadata", None)
                in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
                out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
                tot_tok = getattr(usage, "total_token_count", 0) if usage else (in_tok + out_tok)
                
                return {
                    "text": text_out,
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "retries": retries_used,
                    "status_429": status_429,
                    "status_5xx": status_5xx,
                    "error": None,
                }
            except APIError as e:
                code = getattr(e, "code", 500)
                if code == 429:
                    status_429 += 1
                elif code >= 500:
                    status_5xx += 1
                    
                if attempt == max_retries:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "text": "",
                        "model": model_name,
                        "latency_ms": latency_ms,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "retries": retries_used,
                        "status_429": status_429,
                        "status_5xx": status_5xx,
                        "error": str(e),
                    }
                retries_used += 1
                time.sleep(1.0 * (2 ** attempt) + random.uniform(0.1, 0.4))
            except Exception as e:
                if attempt == max_retries:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "text": "",
                        "model": model_name,
                        "latency_ms": latency_ms,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "retries": retries_used,
                        "status_429": status_429,
                        "status_5xx": status_5xx,
                        "error": str(e),
                    }
                retries_used += 1
                time.sleep(1.0 * (2 ** attempt) + random.uniform(0.1, 0.4))

    def call_structured(
        self,
        prompt: str,
        schema: Any,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Issues structured JSON generation call with fast JSON mode and robust parsing."""
        model_name = model or self.default_model
        # Use response_mime_type without schema constraint to prevent Gemma reasoning hangs
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
        )
        
        t0 = time.perf_counter()
        retries_used = 0
        status_429 = 0
        status_5xx = 0
        
        for attempt in range(max_retries + 1):
            time.sleep(1.0)
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw_text = resp.text if resp and resp.text else "{}"
                
                # Robust json extraction
                parsed_json = {}
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
                cleaned = re.sub(r"\s*```$", "", cleaned)
                try:
                    parsed_json = json.loads(cleaned)
                except Exception:
                    m = re.search(r"(\{.*\})", raw_text, re.DOTALL)
                    if m:
                        try:
                            parsed_json = json.loads(m.group(1))
                        except Exception:
                            parsed_json = {}
                    
                usage = getattr(resp, "usage_metadata", None)
                in_tok = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
                out_tok = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
                tot_tok = (getattr(usage, "total_token_count", 0) or 0) if usage else (in_tok + out_tok)
                
                return {
                    "data": parsed_json,
                    "raw_text": raw_text,
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": tot_tok,
                    "retries": retries_used,
                    "status_429": status_429,
                    "status_5xx": status_5xx,
                    "error": None,
                }
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    status_429 += 1
                elif "500" in err_str or "503" in err_str:
                    status_5xx += 1
                if attempt == max_retries:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "data": {},
                        "raw_text": "",
                        "model": model_name,
                        "latency_ms": latency_ms,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "retries": retries_used,
                        "status_429": status_429,
                        "status_5xx": status_5xx,
                        "error": str(e),
                    }
                retries_used += 1
                time.sleep(2.0 * (2 ** attempt))


def parse_decision_and_citations(
    answer_text: str, candidate_chunks: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses structured decision (ANSWER vs INSUFFICIENT_EVIDENCE vs ERROR)
    and extracts cited chunk citations.
    """
    if not answer_text or not answer_text.strip():
        return "INSUFFICIENT_EVIDENCE", []
        
    text_lower = answer_text.lower()
    
    # Check for explicit refusal signals
    refusal_signals = [
        "insufficient_evidence",
        "insufficient evidence",
        "not contain information",
        "does not contain information",
        "do not specify or mention",
        "does not specify or mention",
        "cannot be determined from the provided",
        "no mention of",
        "not mentioned in the provided",
        "not found in the provided",
        "neither contract mentions",
        "the provided contract does not",
        "the provided excerpts do not",
        "information is not present",
    ]
    
    is_refusal = any(sig in text_lower for sig in refusal_signals)
    decision = "INSUFFICIENT_EVIDENCE" if is_refusal else "ANSWER"
    
    # Extract chunk citations:
    # 1. Match chunk_id patterns (e.g. cuad_contract_001_p0_c0)
    cited_chunk_ids = set()
    for chunk in candidate_chunks:
        cid = chunk["chunk_id"]
        if cid in answer_text:
            cited_chunk_ids.add(cid)
            
    # 2. Match reference indices [Reference 1], [Reference 2], etc.
    ref_matches = re.findall(r'\[(?:Reference|Document Reference|Source Block)\s*(\d+)\]', answer_text, re.IGNORECASE)
    for ref_str in ref_matches:
        try:
            ref_idx = int(ref_str) - 1
            if 0 <= ref_idx < len(candidate_chunks):
                cited_chunk_ids.add(candidate_chunks[ref_idx]["chunk_id"])
        except ValueError:
            pass
            
    # If decision is ANSWER and no explicit citation was formatted in text, default to top-retrieved supporting chunks
    if decision == "ANSWER" and not cited_chunk_ids and candidate_chunks:
        cited_chunk_ids.add(candidate_chunks[0]["chunk_id"])
        
    citations = []
    chunk_map = {c["chunk_id"]: c for c in candidate_chunks}
    for cid in sorted(list(cited_chunk_ids)):
        if cid in chunk_map:
            c = chunk_map[cid]
            citations.append({
                "chunk_id": cid,
                "parent_id": c.get("parent_id"),
                "document_id": c.get("doc_id"),
                "section_path": c.get("section_path", []),
                "supporting_text": c.get("text", "")[:300]
            })
            
    return decision, citations


class Phase6Executor:
    """Executes Real-API RAG pipelines with Layer A/Layer B strict separation."""

    def __init__(self, is_holdout: bool = True):
        self.settings = get_settings()
        self.gateway = RealAPIGatewayClient(api_key=self.settings.gemini_api_key, default_model=self.settings.generation_model)
        self.reranker = LocalCrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, strict=True)
        self.emb_provider = LocalEmbeddingProvider(model_name=cfg.dense_model)
        
        # Load corpus cache
        cache_key = "eeadb154d37e1c13d90ae74e" if is_holdout else "1a9ef6e99dbb234ff50bcd7e"
        self.cache = EvaluationCache(cache_key)
        self.corpus_chunks = self.cache.load_corpus_chunks()
        self.dense_emb, _ = self.cache.load_dense_embeddings()
        
        # Build document-to-chunk indices and BM25 indices
        self.doc_to_chunk_indices: Dict[str, List[int]] = {}
        self.doc_to_bm25: Dict[str, BM25Okapi] = {}
        self.chunk_map = {c["chunk_id"]: c for c in self.corpus_chunks}
        
        for idx, c in enumerate(self.corpus_chunks):
            doc_id = c["doc_id"]
            self.doc_to_chunk_indices.setdefault(doc_id, []).append(idx)
            
        for doc_id, indices in self.doc_to_chunk_indices.items():
            doc_chunks = [self.corpus_chunks[i] for i in indices]
            tokenized = [tokenize_for_bm25(c.get("enriched_text", c["text"])) for c in doc_chunks]
            self.doc_to_bm25[doc_id] = BM25Okapi(tokenized)

    def retrieve_top_k(self, question: str, doc_id: str, top_n: int = 5) -> Tuple[List[Dict[str, Any]], float]:
        """Performs frozen document-scoped hybrid retrieval + TinyBERT rerank."""
        t0 = time.perf_counter()
        doc_indices = self.doc_to_chunk_indices.get(doc_id, [])
        if not doc_indices:
            return [], (time.perf_counter() - t0) * 1000
            
        scoped_chunks = [self.corpus_chunks[i] for i in doc_indices]
        scoped_chunk_ids = [c["chunk_id"] for c in scoped_chunks]
        
        # 1. Dense Search
        q_vec = self.emb_provider.embed_query(question)
        scoped_dense_sims = np.dot(self.dense_emb[doc_indices], q_vec)
        s_dense_top = [scoped_chunk_ids[i] for i in np.argsort(-scoped_dense_sims)[:20]]
        
        # 2. BM25 Search
        bm25_scores = self.doc_to_bm25[doc_id].get_scores(tokenize_for_bm25(question))
        s_bm25_top = [scoped_chunk_ids[i] for i in np.argsort(-bm25_scores)[:20]]
        
        # 3. RRF Fusion
        s_rrf_candidates = [cid for cid, _ in reciprocal_rank_fusion([s_dense_top, s_bm25_top], k=cfg.rrf_k)]
        
        # 4. Parent Chunk Deduplication
        s_dedup = []
        s_p_count: Dict[str, int] = {}
        for cid in s_rrf_candidates:
            c_obj = self.chunk_map.get(cid)
            pid = c_obj.get("parent_id") if c_obj else None
            if pid:
                if s_p_count.get(pid, 0) >= 2:
                    continue
                s_p_count[pid] = s_p_count.get(pid, 0) + 1
            s_dedup.append(cid)
            
        s_budget_20 = s_dedup[:20]
        s_cand_texts = [self.chunk_map[cid]["text"] for cid in s_budget_20]
        
        # 5. TinyBERT CrossEncoder Rerank
        s_rerank_res = self.reranker.rerank(question, s_cand_texts, top_n=top_n)
        final_chunks = [self.chunk_map[s_budget_20[orig_idx]] for orig_idx, _ in s_rerank_res]
        ret_latency_ms = (time.perf_counter() - t0) * 1000
        return final_chunks, ret_latency_ms

    def run_query(self, runtime_payload: Dict[str, Any], variant: str) -> Dict[str, Any]:
        """
        Executes a single query under Layer A (Zero Gold Access).
        variant options: 'BASE_RAG', 'RAG_PLUS_VERIFIER', 'FULL_BOUNDED_MULTI_AGENT'
        """
        query_id = runtime_payload["query_id"]
        question = runtime_payload["question"]
        doc_id = runtime_payload["selected_document_id"]
        
        t_start = time.perf_counter()
        api_trace = []
        planner_res = None
        critic_res = None
        verifier_res = None
        
        # 1. Planner (if FULL)
        t_planner_ms = 0.0
        if variant == "FULL_BOUNDED_MULTI_AGENT":
            p_prompt = f"""You are the Retrieval Planner Agent. Analyze this legal query for document {doc_id}:
Query: "{question}"
Return JSON with fields: {{"complexity": "low", "task_type": "single_contract_qa", "retrieval_strategy": "hybrid_bge_m3_rrf"}}"""
            p_call = self.gateway.call_structured(p_prompt, RetrievalPlan, model=self.settings.planner_model)
            api_trace.append({"component": "planner", **p_call})
            p_data = p_call["data"]
            if isinstance(p_data, list) and p_data:
                p_data = p_data[0]
            planner_res = p_data if isinstance(p_data, dict) else {}
            t_planner_ms = p_call["latency_ms"]

        # 2. Retrieval Execution
        candidates, t_ret_ms = self.retrieve_top_k(question, doc_id, top_n=5)
        
        # 3. Evidence Critic (if FULL)
        t_critic_ms = 0.0
        if variant == "FULL_BOUNDED_MULTI_AGENT" and candidates:
            c_context = "\n\n".join(f"[Snippet {i+1}]: {c['text'][:400]}" for i, c in enumerate(candidates))
            c_prompt = f"""You are the Evidence Critic Agent. Evaluate if these excerpts contain sufficient factual evidence to answer:
Question: "{question}"
Excerpts:
{c_context}
Return JSON with fields: {{"sufficient": true, "recommended_action": "proceed", "expansion_queries": []}}"""
            c_call = self.gateway.call_structured(c_prompt, EvidenceCriticEvaluation, model=self.settings.critic_model)
            api_trace.append({"component": "critic", **c_call})
            c_data = c_call["data"]
            if isinstance(c_data, list) and c_data:
                c_data = c_data[0]
            critic_res = c_data if isinstance(c_data, dict) else {}
            t_critic_ms = c_call["latency_ms"]
            
            # Finite expansion if requested
            if critic_res.get("recommended_action") == "expand_query" and critic_res.get("expansion_queries"):
                exp_q = critic_res["expansion_queries"][0]
                more_cands, _ = self.retrieve_top_k(exp_q, doc_id, top_n=3)
                seen_cids = {c["chunk_id"] for c in candidates}
                for mc in more_cands:
                    if mc["chunk_id"] not in seen_cids:
                        candidates.append(mc)
                        seen_cids.add(mc["chunk_id"])

        # 4. Generator Call
        context_parts = []
        for i, c in enumerate(candidates[:5], 1):
            sec_str = " > ".join(c.get("section_path", [])) if c.get("section_path") else "General"
            context_parts.append(f"[Reference {i}] (Chunk ID: {c['chunk_id']}, Section: {sec_str}):\n{c['text']}")
        context_str = "\n\n---\n\n".join(context_parts)
        
        sys_instruction = """You are a Senior Legal Contract Intelligence Analyst.
Answer the user's question with precise factual accuracy based solely on the provided reference excerpts.
If the reference excerpts do not contain sufficient factual evidence to answer the question, clearly state: "INSUFFICIENT_EVIDENCE: The provided contract excerpts do not contain information to answer this question."
Always cite the exact supporting Reference number and Chunk ID for any factual assertions made (e.g. [Reference 1: <chunk_id>])."""

        gen_prompt = f"""Reference Contract Context:
{context_str}

User Question: {question}

Answer:"""

        g_call = self.gateway.call_text(
            prompt=gen_prompt,
            system_instruction=sys_instruction,
            model=self.settings.generation_model,
            temperature=0.0,
        )
        api_trace.append({"component": "generator", **g_call})
        answer_text = g_call["text"]
        t_gen_ms = g_call["latency_ms"]
        
        decision, citations = parse_decision_and_citations(answer_text, candidates)
        
        # 5. Answer Verifier (if RAG_PLUS_VERIFIER or FULL_BOUNDED_MULTI_AGENT)
        t_ver_ms = 0.0
        if variant in ["RAG_PLUS_VERIFIER", "FULL_BOUNDED_MULTI_AGENT"] and decision == "ANSWER":
            v_context = "\n\n---\n\n".join(f"[Source {i+1}]: {c['text']}" for i, c in enumerate(candidates[:5]))
            v_prompt = f"""You are the Answer Verifier Agent. Audit this generated answer against reference evidence:
Question: "{question}"
Reference Evidence:
{v_context}
Generated Answer:
"{answer_text}"
Return JSON with fields: {{"status": "supported", "recommended_action": "pass", "critique_for_regeneration": ""}}"""
            v_call = self.gateway.call_structured(v_prompt, AnswerVerificationResult, model=self.settings.verifier_model)
            api_trace.append({"component": "verifier", **v_call})
            v_data = v_call["data"]
            if isinstance(v_data, list) and v_data:
                v_data = v_data[0]
            verifier_res = v_data if isinstance(v_data, dict) else {}
            t_ver_ms = v_call["latency_ms"]
            
            # Handle regeneration if requested
            if verifier_res.get("recommended_action") == "regenerate":
                regen_prompt = f"""{gen_prompt}

IMPORTANT CORRECTION: Your previous answer contained ungrounded statements: {verifier_res.get('critique_for_regeneration', 'Check factual grounding')}.
Strictly adhere only to the verbatim factual context provided above."""
                regen_call = self.gateway.call_text(
                    prompt=regen_prompt,
                    system_instruction=sys_instruction,
                    model=self.settings.generation_model,
                    temperature=0.0,
                )
                api_trace.append({"component": "generator_regen", **regen_call})
                if regen_call["text"]:
                    answer_text = regen_call["text"]
                    decision, citations = parse_decision_and_citations(answer_text, candidates)
                    t_gen_ms += regen_call["latency_ms"]
            elif verifier_res.get("recommended_action") == "qualify_or_refuse" or verifier_res.get("status") == "unsupported":
                decision = "INSUFFICIENT_EVIDENCE"
                answer_text = "INSUFFICIENT_EVIDENCE: The provided contract context does not substantiate the facts required to answer this inquiry."

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        
        # Aggregate production usage (excluding offline judge)
        prod_calls = len(api_trace)
        prod_in_tok = sum((c.get("input_tokens") or 0) for c in api_trace)
        prod_out_tok = sum((c.get("output_tokens") or 0) for c in api_trace)
        prod_tot_tok = sum((c.get("total_tokens") or 0) for c in api_trace)
        prod_429 = sum(c.get("status_429", 0) for c in api_trace)
        prod_5xx = sum(c.get("status_5xx", 0) for c in api_trace)
        prod_retries = sum(c.get("retries", 0) for c in api_trace)
        
        return {
            "query_id": query_id,
            "selected_document_id": doc_id,
            "question": question,
            "variant": variant,
            "decision": decision,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunk_ids": [c["chunk_id"] for c in candidates],
            "agent_trace": {
                "planner": planner_res,
                "critic": critic_res,
                "verifier": verifier_res,
            },
            "api_trace": api_trace,
            "telemetry": {
                "total_latency_ms": total_latency_ms,
                "retrieval_latency_ms": t_ret_ms,
                "planner_latency_ms": t_planner_ms,
                "critic_latency_ms": t_critic_ms,
                "generator_latency_ms": t_gen_ms,
                "verifier_latency_ms": t_ver_ms,
                "production_calls": prod_calls,
                "input_tokens": prod_in_tok,
                "output_tokens": prod_out_tok,
                "total_tokens": prod_tot_tok,
                "status_429_count": prod_429,
                "status_5xx_count": prod_5xx,
                "retry_count": prod_retries,
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Real API Evaluator")
    parser.add_argument("--mode", choices=["smoke", "dev-ablation", "final"], required=True, help="Evaluation mode")
    parser.add_argument("--limit", type=int, default=None, help="Optional query limit")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"phase6_{args.mode}_{timestamp}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"PHASE 6 REAL API EVALUATION: MODE={args.mode.upper()}")
    print(f"Run ID: {run_id}")
    print(f"Destination: {run_dir}")
    print("=" * 80)

    # 1. Load Manifest
    if args.mode in ["smoke", "dev-ablation"]:
        manifest_path = REPO_ROOT / "evaluation" / "manifests" / "phase6_dev_api_manifest.json"
        is_holdout = False
    else:
        manifest_path = REPO_ROOT / "evaluation" / "manifests" / "phase6_final_api_manifest.json"
        is_holdout = True

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_queries = manifest_data["queries"]
    
    if args.mode == "smoke":
        # 10 answerable, 10 unanswerable
        ans = [q for q in raw_queries if not q.get("is_unanswerable", False)][:10]
        unans = [q for q in raw_queries if q.get("is_unanswerable", False)][:10]
        eval_queries = ans + unans
        variants_to_run = ["BASE_RAG"]
    elif args.mode == "dev-ablation":
        eval_queries = raw_queries if not args.limit else raw_queries[:args.limit]
        variants_to_run = ["BASE_RAG", "RAG_PLUS_VERIFIER", "FULL_BOUNDED_MULTI_AGENT"]
    else: # final
        eval_queries = raw_queries if not args.limit else raw_queries[:args.limit]
        variants_to_run = ["FULL_BOUNDED_MULTI_AGENT"]

    print(f"Total Evaluation Queries: {len(eval_queries)} (Variants: {variants_to_run})")
    
    # Save manifest snapshot in run directory
    (run_dir / "manifest_snapshot.json").write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    executor = Phase6Executor(is_holdout=is_holdout)

    predictions_file = run_dir / "predictions.jsonl"
    all_predictions = []

    # 2. Execute Real API Workload
    with open(predictions_file, "w", encoding="utf-8") as f_pred:
        for v_idx, variant in enumerate(variants_to_run):
            print(f"\n--- Running Variant [{v_idx+1}/{len(variants_to_run)}]: {variant} ---")
            for q_idx, q in enumerate(eval_queries):
                # LAYER A SANITIZED RUNTIME PAYLOAD: ZERO GOLD ACCESS
                runtime_payload = {
                    "query_id": q["query_id"],
                    "question": q["question"],
                    "selected_document_id": q["source_contract_id"]
                }
                
                res = executor.run_query(runtime_payload, variant=variant)
                all_predictions.append(res)
                f_pred.write(json.dumps(res) + "\n")
                f_pred.flush()
                
                decision_sym = "[ANS]" if res["decision"] == "ANSWER" else "[REF]"
                tel = res["telemetry"]
                q_text_snippet = q["question"][:50].encode("ascii", "replace").decode("ascii")
                print(f"  [{q_idx+1:03d}/{len(eval_queries):03d}] {decision_sym} | {tel['total_latency_ms']:.0f}ms | {tel['production_calls']} calls | {tel['total_tokens']} tok | Q: {q_text_snippet}...")

    # 3. Compute Run SHA-256 Hash
    pred_bytes = predictions_file.read_bytes()
    pred_hash = hashlib.sha256(pred_bytes).hexdigest()
    print(f"\n[OK] Predictions saved and frozen: {predictions_file.name} (SHA256: {pred_hash})")

    # 4. Save Run Config and Summary
    run_config = {
        "run_id": run_id,
        "mode": args.mode,
        "is_holdout": is_holdout,
        "manifest_path": str(manifest_path),
        "generator_model": settings.generation_model,
        "planner_model": settings.planner_model,
        "critic_model": settings.critic_model,
        "verifier_model": settings.verifier_model,
        "retrieval_protocol": "v4.2.0_frozen",
        "variants_evaluated": variants_to_run,
        "total_predictions": len(all_predictions),
        "predictions_sha256": pred_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"[OK] Run config saved to {run_dir / 'run_config.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
