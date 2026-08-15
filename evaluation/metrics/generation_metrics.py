"""
Generation Evaluation Metrics.
Implements Token F1, Exact Match (EM), Grounded Faithfulness, and Refusal Accuracy.
"""
import re
import string
from typing import List, Dict, Any, Optional


def normalize_text(s: str) -> str:
    """Lower text, remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact Match (EM) metric."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens).intersection(set(gt_tokens))
    num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_faithfulness(prediction: str, context_chunks: List[str]) -> float:
    """
    Computes grounded faithfulness ratio of generated answer sentences against context.
    """
    if not prediction.strip():
        return 0.0
    if not context_chunks:
        return 0.0

    combined_context = " ".join(context_chunks).lower()
    sentences = [s.strip() for s in re.split(r"[.!?]", prediction) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    supported_count = 0
    for sent in sentences:
        words = [w for w in normalize_text(sent).split() if len(w) > 3]
        if not words:
            supported_count += 1
            continue
        
        # Word overlap check with context
        overlap = sum(1 for w in words if w in combined_context)
        if (overlap / len(words)) >= 0.50:
            supported_count += 1

    return supported_count / len(sentences)


def evaluate_refusal_accuracy(prediction: str, is_unanswerable: bool) -> float:
    """
    Measures if the system correctly refused an unanswerable question or answered an answerable one.
    """
    refusal_keywords = [
        "not specified", "does not contain", "cannot determine", "không có thông tin",
        "tài liệu không đề cập", "no reference", "insufficient evidence"
    ]
    pred_lower = prediction.lower()
    is_refusal = any(k in pred_lower for k in refusal_keywords)

    if is_unanswerable:
        return 1.0 if is_refusal else 0.0
    else:
        return 1.0 if not is_refusal else 0.0
