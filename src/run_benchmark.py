"""
Run the HumanEvalFix benchmark with the agent.
"""

import argparse
import os
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
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (slower but potentially more accurate)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Check environment variable for thinking mode (can be overridden by CLI flag)
    env_thinking = os.getenv('ENABLE_THINKING', 'false').lower() in ('true', '1', 'yes')
    enable_thinking = args.thinking or env_thinking
    
    # Validate arguments
    if args.full and args.partial:
        print("Error: Cannot use both --full and --partial flags")
        return
    
    if args.partial is not None and (args.partial < 1 or args.partial > 100):
        print("Error: --partial must be between 1 and 100")
        return
    
    if enable_thinking:
        print("+++ Thinking mode ENABLED - expect slower but potentially more accurate results +++")
        print("+++ Parameters: temperature=0.6, top_p=0.95, top_k=20 +++\n")
    else:
        print("+++ Thinking mode DISABLED - faster inference (default) +++")
        print("+++ Parameters: temperature=0.7, top_p=0.8, top_k=20 +++\n")
    
    print("Loading HumanEvalPack dataset...")
    
    ds = load_dataset("bigcode/humanevalpack", "python")["test"]
    
    print(f"Dataset loaded: {len(ds)} problems")
    print(f"Dataset fields: {list(ds[0].keys())}")
    
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
    
    result = run_benchmark(
        dataset=ds,
        max_problems=max_problems,
        verbose=args.debug,
        enable_thinking=enable_thinking
    )
    
    # Save to results directory if it exists (for Docker), otherwise current dir
    output_dir = "results" if os.path.exists("results") else "."
    output_path = os.path.join(output_dir, "benchmark_results.json")
    save_results(result, output_path)

if __name__ == "__main__":
    main()

