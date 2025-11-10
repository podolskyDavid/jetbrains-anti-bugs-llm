import os
import re
import subprocess
import tempfile
from models import AgentInput
from typing import Literal, TypedDict, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Configuration
MODEL_NAME = os.getenv('OLLAMA_MODEL', 'hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q8_K_XL')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
MAX_ITERATIONS = 3
TEST_TIMEOUT = 30

# Qwen3 runs in 2 modes: thinking and non-thinking. Unrelated to ReAct.
def get_thinking_directive(enable_thinking: bool) -> str:
    """Get the thinking mode directive for Qwen3 prompts."""
    return "" if enable_thinking else " /no_think"


def get_llm_params(enable_thinking: bool) -> dict:
    """
    Get LLM parameters based on thinking mode.
    Official Qwen3 recommendations:
    - Non-thinking: temp=0.7, top_p=0.8, top_k=20
    - Thinking: temp=0.6, top_p=0.95, top_k=20
    """
    if enable_thinking:
        return {
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 20,
        }
    else:
        return {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
        }


class AgentState(TypedDict):
    """State for the ReAct agent."""
    buggy_solution: str
    test: str
    entry_point: str
    iteration: int
    fixed_solution: str
    last_fix_attempt: str
    test_result: Optional[str]
    test_passed: bool


def extract_function_name(code: str) -> Optional[str]:
    """Extract function name from Python code."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def run_code_tests(code: str, test: str, entry_point: str, timeout: int = TEST_TIMEOUT) -> tuple[bool, str]:
    """
    Run tests on code in sandboxed environment.
    Returns (passed: bool, message: str)
    """
    # Combine code and tests
    full_code = f"{code}\n\n{test}"
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        # Run in subprocess with timeout
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return True, "All tests passed"
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Tests failed: {error_msg[:500]}"
            
    except subprocess.TimeoutExpired:
        return False, f"Execution timeout after {timeout} seconds"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def extract_code_block(text: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code in markdown blocks
    code_block_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()  # Take last block
    
    # If no markdown blocks, try to extract function definition
    lines = text.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
        if in_function:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    return text.strip()


FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Buggy Code:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

Thought: The loop checks every number from 2 to n-1. This is inefficient - we only need to check up to sqrt(n) because factors come in pairs.

Action:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

EXAMPLE 2:
Buggy Code:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n + 1)

Thought: The recursive call uses n+1, which increases n instead of decreasing it. This causes infinite recursion. Should be n-1 to reach base case.

Action:
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

EXAMPLE 3:
Buggy Code:
def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val

Thought: Initializing max_val to 0 is wrong - if all numbers are negative, it returns 0 instead of the actual maximum. Should initialize to first element.

Action:
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
"""


def create_react_graph(enable_thinking: bool, verbose: bool = False) -> StateGraph:
    """Create a ReAct-style LangGraph for fixing buggy code."""
    
    llm_params = get_llm_params(enable_thinking)
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        **llm_params
    )
    
    def analyze_and_fix_node(state: AgentState) -> AgentState:
        """ReAct: Thought + Action (generate fix)."""
        thinking_directive = get_thinking_directive(enable_thinking)
        iteration = state['iteration']
        
        # Build context from previous attempts using ReAct observation 
        context = ""
        if iteration > 0 and state.get('test_result'):
            context = f"""
OBSERVATION (from previous attempt):
{state['test_result']}

Your previous action:
{state['last_fix_attempt']}

The test failed. Analyze why and try a different approach.
"""
        
        # Adjust output instructions based on thinking mode
        if enable_thinking:
            output_instruction = """Follow the ReAct pattern:

Thought: [Identify the bug and plan your fix - this will be in your internal reasoning]

Action (output only the fixed code):
```python
def {entry_point}(...):
    # fixed implementation
    pass
```"""
        else:
            output_instruction = """Follow the ReAct pattern:

Thought: [Identify the bug and plan your fix]

Action:
```python
def {entry_point}(...):
    # fixed implementation
    pass
```"""
        
        prompt = f"""You are an expert Python debugger using the ReAct (Reasoning + Acting) approach.

{FEW_SHOT_EXAMPLES}

CURRENT TASK:
Buggy Code:
{state['buggy_solution']}

Tests (must all pass):
{state['test']}

{context}

{output_instruction.format(entry_point=state['entry_point'])}

Remember: Keep function name as {state['entry_point']}, make minimal changes.{thinking_directive}"""
        
        if verbose and iteration == 0:
            print("+++ REACT ITERATION 1: THOUGHT + ACTION +++\n")
        elif verbose:
            print(f"+++ REACT ITERATION {iteration + 1}: THOUGHT + ACTION (after observation) +++\n")
        
        full_response = ""
        for chunk in llm.stream(prompt):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                if verbose:
                    print(content, end="", flush=True)
        
        if verbose:
            print("\n")
        
        fixed_code = extract_code_block(full_response)
        
        return {
            "last_fix_attempt": fixed_code,
            "iteration": iteration + 1
        }
    
    def test_code_node(state: AgentState) -> AgentState:
        """ReAct: Observation - test the fix and observe results."""
        if verbose:
            print("+++ REACT OBSERVATION: TESTING THE ACTION +++\n")
        
        passed, message = run_code_tests(
            state['last_fix_attempt'],
            state['test'],
            state['entry_point']
        )
        
        if verbose:
            if passed:
                print(f"Observation: TESTS PASSED ✓\n")
            else:
                print(f"Observation: TESTS FAILED\n{message}\n")
        
        # +++ If tests pass, this is our fixed solution
        # +++ If all attempts exhausted, return last attempt
        return {
            "test_passed": passed,
            "test_result": message,
            "fixed_solution": state['last_fix_attempt']
        }
    
    def route_after_test(state: AgentState) -> Literal["analyze", "end"]:
        """Route after Observation: continue ReAct loop or end."""
        # +++ Observation shows success - we're done!
        if state['test_passed']:
            return "end"
        # +++ Max ReAct iterations reached - return best attempt
        elif state['iteration'] >= MAX_ITERATIONS:
            return "end"
        # +++ Continue ReAct cycle: Observation → Thought + Action
        else:
            return "analyze"
    
    # Build graph - Pure ReAct: Thought+Action → Observation → loop
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("analyze_and_fix", analyze_and_fix_node)
    graph.add_node("test_code", test_code_node)
    
    # Add edges
    graph.add_edge(START, "analyze_and_fix")
    graph.add_edge("analyze_and_fix", "test_code")
    graph.add_conditional_edges(
        "test_code",
        route_after_test,
        {
            "analyze": "analyze_and_fix",  # Continue ReAct loop
            "end": END                      # Tests passed or max iterations
        }
    )
    
    return graph.compile()


def fix_code_agent(
    agent_input: AgentInput,
    enable_thinking: bool,
    verbose: bool = False,
    bug_type: str = None
) -> str:
    """
    Main entry point: Fix buggy Python code using ReAct agent.
    
    Args:
        agent_input: The agent input with buggy code and tests
        enable_thinking: Whether to enable thinking mode
        verbose: Whether to enable debug output
        bug_type: Bug type for debugging (optional)
        
    Returns:
        Fixed code
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"+++ FIX CODE AGENT START +++")
        print(f"Task: {agent_input.task_id}")
        print(f"Bug type: {bug_type}")
        print(f"\nBuggy Code:")
        print(f"{'-'*80}")
        print(f"{agent_input.buggy_solution}")
        print(f"{'-'*80}")
        print(f"\nTest Code:")
        print(f"{'-'*80}")
        print(f"{agent_input.test}")
        print(f"{'='*80}\n")
    
    # Extract entry point (function name) from test or buggy code
    entry_point = agent_input.entry_point or extract_function_name(agent_input.buggy_solution)
    if not entry_point:
        # Try to extract from test
        test_match = re.search(r'def\s+check\s*\(\s*(\w+)\s*\)', agent_input.test)
        if test_match:
            entry_point = test_match.group(1)
        else:
            entry_point = "unknown_function"
    
    # Create and run the ReAct graph
    graph = create_react_graph(enable_thinking, verbose=verbose)
    
    initial_state: AgentState = {
        "buggy_solution": agent_input.buggy_solution,
        "test": agent_input.test,
        "entry_point": entry_point,
        "iteration": 0,
        "fixed_solution": "",
        "last_fix_attempt": "",
        "test_result": None,
        "test_passed": False
    }
    
    result = graph.invoke(initial_state)
    
    if verbose:
        print(f"\n{'='*80}")
        if result['test_passed']:
            print("+++ REACT CYCLE COMPLETE: SUCCESS +++")
        else:
            print(f"+++ REACT CYCLE STOPPED: {MAX_ITERATIONS} iterations exhausted +++")
            print("+++ Returning last action +++")
        print(f"{'='*80}\n")
    
    return result["fixed_solution"]