"""
Helper CLI Script to run the 7-Variant Adaptive Pipeline Ablation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.eval_ablation import AblationBenchmarkRunner, format_ablation_markdown_table

if __name__ == "__main__":
    runner = AblationBenchmarkRunner()
    results = runner.run_all_variants()
    print("\n" + format_ablation_markdown_table(results))
