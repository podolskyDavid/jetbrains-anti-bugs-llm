import time
import os
import sys
import json
import traceback
from io import StringIO
from typing import Optional
from contextlib import redirect_stdout, redirect_stderr

from models import ProblemData, AgentInput, ProblemOutput, BenchmarkResult
from agent import fix_code_agent


def execute_code_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
) -> tuple[bool, Optional[str]]:
    """
    Execute the fixed code with test cases in a sandboxed environment.
    
    Args:
        code: The fixed code to test
        test_code: The test cases to run
        entry_point: The function name being tested
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
        - success: True if all tests passed, False otherwise
        - error_message: Error details if tests failed, None if passed
    """
    # Create isolated namespace for execution
    namespace = {
        '__builtins__': __builtins__,
        'sys': sys,
    }
    
    try:
        # Capture stdout and stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the fixed code
            exec(code, namespace)
            
            # Verify the entry point exists
            if entry_point not in namespace:
                return False, f"Entry point '{entry_point}' not found in fixed code"
            
            # Execute the tests
            exec(test_code, namespace)
            
        # If we get here, all tests passed
        return True, None
        
    except AssertionError as e:
        # Test assertion failed
        error_msg = f"Test assertion failed: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg
        
    except SyntaxError as e:
        # Syntax error in code
        error_msg = f"Syntax error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg
        
    except Exception as e:
        # Other runtime errors
        error_msg = f"Runtime error: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg


def evaluate_single_problem(problem: ProblemData, verbose: bool = True, enable_thinking: bool = False) -> ProblemOutput:
    """
    Evaluate a single problem: get agent's fix and test it.
    
    Args:
        problem: Input problem data (ProblemData from dataset)
        verbose: Whether to print debug output
        
    Returns:
        ProblemOutput with results
    """
    start_time = time.time()
    
    try:
        # Get the agent's fixed solution (will print streaming output in debug mode)
        
        fixed_solution: str = fix_code_agent(problem.to_agent_input(), enable_thinking=enable_thinking, verbose=verbose, bug_type=problem.bug_type)
        
        # Execute and test the fixed code
        if verbose:
            print(f"\n{'='*80}")
            print(f"+++ TESTING FIXED CODE +++")
        
        passed, error_message = execute_code_with_tests(
            code=fixed_solution,
            test_code=problem.test,
            entry_point=problem.entry_point,
        )
        
        execution_time = time.time() - start_time
        
        # Print result
        if verbose:
            if passed:
                print(f"+++ TEST PASSED (time: {execution_time:.2f}s)")
            else:
                print(f"!!! TEST FAILED (time: {execution_time:.2f}s)")
                print(f"\n!!! ERROR MESSAGE:")
                print(error_message)
            print(f"\n{'='*80}\n")
        
        return ProblemOutput(
            task_id=problem.task_id,
            fixed_solution=fixed_solution,
            passed=passed,
            error_message=error_message,
            execution_time=execution_time
        )
        
    except Exception as e:
        # Handle any unexpected errors in the pipeline
        execution_time = time.time() - start_time
        error_msg = f"Pipeline error: {str(e)}\n{traceback.format_exc()}"
        
        if verbose:
            print(f"!!! PIPELINE ERROR:")
            print(error_msg)
            print(f"\n{'='*80}\n")
        
        return ProblemOutput(
            task_id=problem.task_id,
            fixed_solution="",
            passed=False,
            error_message=error_msg,
            execution_time=execution_time
        )


def run_benchmark(
    dataset,
    max_problems: Optional[int] = None,
    verbose: bool = True,
    enable_thinking: bool = False
) -> BenchmarkResult:
    """
    Main benchmark loop: iterate through problems and evaluate each one.
    
    Args:
        dataset: HumanEvalFix dataset (e.g., from datasets.load_dataset)
        max_problems: Optional limit on number of problems to evaluate
        verbose: Whether to print progress
        
    Returns:
        BenchmarkResult with overall statistics and individual results
    """
    benchmark_result = BenchmarkResult(
        total_problems=0,
        passed_problems=0,
        failed_problems=0,
        pass_at_1=0.0,
        results=[]
    )
    
    # Determine how many problems to evaluate
    num_problems = len(dataset) if max_problems is None else min(max_problems, len(dataset))
    
    if verbose:
        print(f"Starting benchmark evaluation on {num_problems} problems...")
        print(f"{'='*80}")
    
    for idx, row in enumerate(dataset):
        if max_problems is not None and idx >= max_problems:
            break
            
        # Convert dataset row to ProblemData
        problem = ProblemData(
            task_id=row.get('task_id', f'problem_{idx}'),
            prompt=row['prompt'],
            buggy_solution=row['buggy_solution'],
            canonical_solution=row['canonical_solution'],
            test=row['test'],
            entry_point=row['entry_point'],

            import_=row.get('import'),
            test_setup=row.get('test_setup'),

            declaration=row.get('declaration'),
            bug_type=row.get('bug_type'),
            failure_symptoms=row.get('failure_symptoms'),
            signature=row.get('signature'),
            docstring=row.get('docstring'),
            instruction=row.get('instruction'),
            example_test=row.get('example_test')
        )
        
        if verbose:
            print(f"\n[{idx + 1}/{num_problems}] Evaluating problem: {problem.task_id}")
        
        # Evaluate the problem
        result = evaluate_single_problem(problem, verbose=verbose, enable_thinking=enable_thinking)
        benchmark_result.add_result(result)
        
        if verbose and (idx + 1) % 10 == 0:
            print(f"\n--- Progress: {idx + 1}/{num_problems} ---")
            print(f"  Current pass@1: {benchmark_result.pass_at_1:.2%}")
            print(f"  Passed: {benchmark_result.passed_problems}")
            print(f"  Failed: {benchmark_result.failed_problems}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print("=" * 80)
        print(f"Total problems: {benchmark_result.total_problems}")
        print(f"Passed: {benchmark_result.passed_problems}")
        print(f"Failed: {benchmark_result.failed_problems}")
        print(f"pass@1: {benchmark_result.pass_at_1:.2%}")
        print("=" * 80)
    
    return benchmark_result


def save_results(result: BenchmarkResult, output_path: str):
    """
    Save benchmark results to a JSON file.
    
    Args:
        result: BenchmarkResult to save
        output_path: Path to output JSON file
    """

    with open(output_path, 'w') as f:
        json.dump(result.model_dump(), f, indent=2)
    
    print(f"Results saved to: {output_path}")

