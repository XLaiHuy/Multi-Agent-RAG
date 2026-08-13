import sys
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

def test_real_domain_datasets():
    # 1. Financial QA 10-K (Corporate Annual Financial Reports & SEC Filings)
    try:
        ds_fin = load_dataset("virattt/financial-qa-10K", split="train[:50]")
        print("✅ [1/2] Financial QA 10-K loaded:", len(ds_fin))
        print("   Sample question:", ds_fin[0].get("question"))
        print("   Sample context:", ds_fin[0].get("context")[:120] if ds_fin[0].get("context") else ds_fin[0].get("answer")[:120])
    except Exception as e:
        print("❌ [1/2] Financial QA 10-K error:", e)

    # 2. Vietnamese Legal Documents
    try:
        ds_legal = load_dataset("th1nhng0/vietnamese-legal-documents", split="train[:50]")
        print("✅ [2/2] Vietnamese Legal Documents loaded:", len(ds_legal))
        print("   Sample keys:", ds_legal[0].keys())
    except Exception as e:
        print("  Trying Legal RAG Bench...")
        try:
            ds_legal = load_dataset("isaacus/legal-rag-bench", split="test[:50]")
            print("✅ [2/2] Legal RAG Bench loaded:", len(ds_legal))
        except Exception as e2:
            print("❌ [2/2] Legal dataset error:", e2)

if __name__ == "__main__":
    test_real_domain_datasets()
