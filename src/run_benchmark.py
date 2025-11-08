"""
Run the HumanEvalFix benchmark with the agent.
"""

import argparse
import random
from datasets import load_dataset
from pipeline import run_benchmark, save_results


def main():
    """
    Load HumanEvalPack dataset and run the benchmark.
    """
    parser = argparse.ArgumentParser(
        description="Run the HumanEvalFix benchmark with the agent."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests in the dataset"
    )
    parser.add_argument(
        "--partial",
        type=int,
        metavar="PERCENT",
        help="Run a random percentage of tests (e.g., --partial 33 for 33%%)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.full and args.partial:
        print("Error: Cannot use both --full and --partial flags")
        return
    
    if args.partial is not None and (args.partial < 1 or args.partial > 100):
        print("Error: --partial must be between 1 and 100")
        return
    
    print("Loading HumanEvalPack dataset...")
    
    ds = load_dataset("bigcode/humanevalpack", "python")["test"]
    
    print(f"Dataset loaded: {len(ds)} problems")
    print("\nExample dataset fields:")
    row = ds[0]
    print(f"  Available keys: {list(row.keys())}")
    
    # Determine how many problems to run
    if args.full:
        max_problems = len(ds)
        print(f"\n[Mode: FULL] Running all {max_problems} problems")
    elif args.partial:
        max_problems = max(1, int(len(ds) * args.partial / 100))
        print(f"\n[Mode: PARTIAL {args.partial}%] Running {max_problems} randomly selected problems")
        
        # Randomly sample indices
        all_indices = list(range(len(ds)))
        random.shuffle(all_indices)
        selected_indices = all_indices[:max_problems]
        ds = ds.select(selected_indices)
    else:
        max_problems = 5
        print(f"\n[Mode: DEFAULT] Running {max_problems} problems (use --full or --partial to change)")
    
    print("\n" + "=" * 80)
    print("Running benchmark with LLM agent...")
    print("=" * 80)
    
    result = run_benchmark(
        dataset=ds,
        max_problems=max_problems,
        verbose=True
    )
    
    save_results(result, "benchmark_results.json")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"pass@1: {result.pass_at_1:.2%}")
    print(f"Total: {result.total_problems}")
    print(f"Passed: {result.passed_problems}")
    print(f"Failed: {result.failed_problems}")


if __name__ == "__main__":
    main()

