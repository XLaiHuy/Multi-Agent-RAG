import sys
from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding="utf-8")

api = HfApi()

print("Searching HuggingFace Hub for 'finance' datasets...")
fin_datasets = api.list_datasets(search="financial", limit=10)
for d in fin_datasets:
    print(f"  • {d.id}")

print("\nSearching HuggingFace Hub for 'legal' datasets...")
legal_datasets = api.list_datasets(search="legal", limit=10)
for d in legal_datasets:
    print(f"  • {d.id}")
