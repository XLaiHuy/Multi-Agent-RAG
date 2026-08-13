import sys
from datasets import load_dataset

sys.stdout.reconfigure(encoding="utf-8")

def test_parquet_datasets():
    # Financial QA
    try:
        ds = load_dataset("ebiac/financial-qa", split="train[:20]")
        print("✅ [1/2] ebiac/financial-qa loaded successfully!", len(ds))
        print("Sample:", ds[0])
    except Exception as e:
        print("❌ [1/2] ebiac/financial-qa error:", e)

    # Legal QA / ViMRC
    try:
        ds_legal = load_dataset("nguyendac/vietnamese-legal-qa", split="train[:20]")
        print("✅ [2/2] nguyendac/vietnamese-legal-qa loaded successfully!", len(ds_legal))
        print("Sample:", ds_legal[0])
    except Exception as e:
        print("  Trying alternative legal dataset: bkai-foundation-models/vi-mrc-large...")
        try:
            ds_legal = load_dataset("bkai-foundation-models/vi-mrc-large", split="train[:20]")
            print("✅ [2/2] bkai-foundation-models/vi-mrc-large loaded!", len(ds_legal))
            print("Sample:", ds_legal[0])
        except Exception as e2:
            print("❌ [2/2] Legal QA error:", e2)

if __name__ == "__main__":
    test_parquet_datasets()
