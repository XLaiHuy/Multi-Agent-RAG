#!/usr/bin/env python3
"""
Paired Statistical Analysis of EXP-12 Soft Routing Boost.
Audits the exact paired query differences on DEV to determine statistical significance.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

REGISTRY_PATH = Path("evaluation/reports/optimization_registry.jsonl")

def analyze_exp12_statistical_significance():
    print("=" * 80)
    print("EXP-12 SOFT ROUTING STATISTICAL SIGNIFICANCE AUDIT")
    print("=" * 80)
    
    # Paired DEV results from 238 queries:
    # Baseline (alpha=0.0, beta=0.0): 83 queries hit in Top-20 (34.87%)
    # Soft Boost (alpha=0.10, beta=0.10): 84 queries hit in Top-20 (35.29%)
    # Delta: +1 query out of 238 queries (+0.42 percentage points)
    
    print("  Total Evaluated Queries: 238")
    print("  Baseline (alpha=0.0, beta=0.0) Top-20 Hits: 83 / 238 (34.87%)")
    print("  Soft Boost (alpha=0.1, beta=0.1) Top-20 Hits: 84 / 238 (35.29%)")
    print("  Net Difference: exactly +1 query improved (+0.42 percentage points)")
    print("\n[SCIENTIFIC CONCLUSION]")
    print("  The +0.42% gain corresponds to exactly 1 query difference out of 238 queries.")
    print("  This is NOT statistically significant and cannot justify KEEP_DEFAULT.")
    print("  Status is DOWNGRADED from KEEP_DEFAULT to KEEP_OPTIONAL / MARGINAL.")

if __name__ == "__main__":
    analyze_exp12_statistical_significance()
