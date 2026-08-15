"""
Helper CLI Script to run the Full CUAD Benchmark Suite.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmarks.run_cuad_benchmark import run_full_cuad_benchmark

if __name__ == "__main__":
    report = run_full_cuad_benchmark()
