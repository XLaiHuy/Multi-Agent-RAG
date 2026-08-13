"""
Script to test loading real public Finance, Legal, and Document OCR datasets from HuggingFace Hub:
1. UIT-ViQuAD 2.0 (Vietnamese QA)
2. Stanford SQuAD 2.0 (Global QA)
3. Financial QA / FinQA (Real Public Corporate Financial Reports & Audits)
4. CUAD (Contract Understanding Atticus Dataset - Real Commercial Legal Contracts)
"""
import sys
from pathlib import Path
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

def test_hf_datasets():
    print("Testing HuggingFace Public Real Datasets...")
    
    # 1. UIT-ViQuAD 2.0
    try:
        ds_viquad = load_dataset("taidng/UIT-ViQuAD2.0", split="validation[:10]")
        print(f"✅ [1/4] UIT-ViQuAD 2.0 loaded: {len(ds_viquad)} samples.")
    except Exception as e:
        print(f"❌ [1/4] UIT-ViQuAD 2.0 failed: {e}")

    # 2. SQuAD 2.0
    try:
        ds_squad = load_dataset("rajpurkar/squad_v2", split="validation[:10]")
        print(f"✅ [2/4] SQuAD 2.0 loaded: {len(ds_squad)} samples.")
    except Exception as e:
        print(f"❌ [2/4] SQuAD 2.0 failed: {e}")

    # 3. Financial QA (Real Corporate Reports)
    try:
        ds_fin = load_dataset("virat/financial-qa", split="train[:10]")
        print(f"✅ [3/4] Financial-QA loaded: {len(ds_fin)} samples.")
    except Exception as e:
        print(f"  [3/4] Trying alternative Financial dataset (Financial Phrasebank)...")
        try:
            ds_fin = load_dataset("financial_phrasebank", "sentences_allagree", split="train[:10]")
            print(f"✅ [3/4] Financial Phrasebank loaded: {len(ds_fin)} samples.")
        except Exception as e2:
            print(f"❌ [3/4] Financial dataset error: {e2}")

    # 4. CUAD Legal Contracts (Commercial Law)
    try:
        ds_legal = load_dataset("cuad", split="test[:10]")
        print(f"✅ [4/4] CUAD Legal Contracts loaded: {len(ds_legal)} samples.")
    except Exception as e:
        print(f"❌ [4/4] Legal CUAD failed: {e}")

if __name__ == "__main__":
    test_hf_datasets()
