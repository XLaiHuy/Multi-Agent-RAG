from datasets import load_dataset

ds = load_dataset("isaacus/legal-rag-bench", split="test[:5]")
print("Legal RAG Bench keys:", ds[0].keys())
print("Legal sample:", list(ds[0].items())[:3])
