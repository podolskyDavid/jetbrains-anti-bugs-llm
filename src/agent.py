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
TEST_TIMEOUT = 5

# +++ LLM Token/Repetition Limits
MAX_OUTPUT_TOKENS = 1200      # Maximum tokens per LLM response
REPEAT_LAST_N = 32            # Look back N tokens for repetition detection
REPEAT_PENALTY = 1.3          # Penalty for repeating tokens (1.0 = none, higher = stronger)

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
<examples>

<example id="1" topic="analyzing_tests_for_patterns">
<buggy_code>
def count_upper(s):
    count = 0
    for i in range(0, len(s), 2):
        if s[i] in "AEIOU":
            count += 2
    return count
</buggy_code>

<tests>
assert count_upper('aBCdEf') == 1
assert count_upper('dBBE') == 0
assert count_upper('EEEE') == 2
</tests>

<thought>
Let me analyze the test cases carefully:
- 'aBCdEf': Has E at index 4. Test expects 1.
- 'dBBE': Has E at index 3. Test expects 0.
- 'EEEE': Has E at indices 0,1,2,3. Test expects 2.

Pattern: Only positions at EVEN indices (0,2,4...) count! Looking at 'dBBE', E is at index 3 (odd) → 0. Looking at 'EEEE', E at indices 0,2 → count is 2. 

The buggy code has TWO errors: (1) counts by 2 instead of 1, (2) returns count*2. Fix: check even indices, count by 1.
</thought>

<action>
```python
def count_upper(s):
    count = 0
    for i in range(0, len(s), 2):
        if s[i] in "AEIOU":
            count += 1
    return count
```
</action>
</example>

<example id="2" topic="understanding_recursion_from_tests">
<buggy_code>
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n + 1)
</buggy_code>

<tests>
assert factorial(0) == 1
assert factorial(3) == 6
</tests>

<thought>
The recursive call uses n+1, which increases n instead of decreasing it. This causes infinite recursion because we never reach the base case n==0. Should be n-1 to count down to base case.
</thought>

<action>
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```
</action>
</example>

<example id="3" topic="edge_cases_reveal_bugs">
<buggy_code>
def find_max(numbers):
    max_val = 0
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
</buggy_code>

<tests>
assert find_max([1, 5, 3]) == 5
assert find_max([-10, -5, -20]) == -5
</tests>

<thought>
Test with all negative numbers reveals the bug! Initializing max_val to 0 fails when all numbers are negative (-10, -5, -20 are all < 0), so it returns 0 instead of -5. Must initialize to first element or negative infinity.
</thought>

<action>
```python
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
```
</action>
</example>

</examples>
"""


def create_react_graph(enable_thinking: bool, verbose: bool = False) -> StateGraph:
    """Create a ReAct-style LangGraph for fixing buggy code."""
    
    llm_params = get_llm_params(enable_thinking)
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        num_predict=MAX_OUTPUT_TOKENS,   # +++ Max tokens to generate - prevents infinite loops
        repeat_last_n=REPEAT_LAST_N,     # +++ Look back N tokens for repetition detection
        repeat_penalty=REPEAT_PENALTY,   # +++ Penalize repetition (1.0 = no penalty, higher = stronger)
        **llm_params
    )
    
    def analyze_and_fix_node(state: AgentState) -> AgentState:
        """ReAct: Thought + Action (generate fix)."""
        thinking_directive = get_thinking_directive(enable_thinking)
        iteration = state['iteration']
        
        # +++ Calculate remaining attempts dynamically
        remaining_attempts = MAX_ITERATIONS - iteration
        if remaining_attempts == 1:
            attempts_message = "You have ONLY 1 attempt left to fix the code"
        else:
            attempts_message = f"You have {remaining_attempts} attempts remaining"
        
        # Build context from previous attempts using ReAct observation 
        context = ""
        if iteration > 0 and state.get('test_result'):
            context = f"""
<observation>
Your previous fix FAILED. Test output:
<test_output>
{state['test_result']}
</test_output>

<previous_action>
{state['last_fix_attempt']}
</previous_action>
</observation>

<reflection>
Your previous fix FAILED. You must analyze WHY before trying again.

STOP and manually calculate the expected output:
1. Take the FIRST test case: work through it step-by-step BY HAND
2. What value does the test expect? Calculate it manually
3. What value did your code produce? Why is it different?
4. What assumption did you make that was WRONG?
5. CHECK THE FUNCTION SIGNATURE: What parameters does the function take? Are you using ALL of them correctly?

You MUST try a COMPLETELY DIFFERENT approach. Do NOT repeat the same logic!
</reflection>
"""
        
        # Adjust output instructions based on thinking mode
        if enable_thinking:
            output_instruction = """<instructions>
Provide your response following the ReAct pattern:

<thought>
[Analyze the test cases carefully to understand requirements, identify the bug, and plan your fix - this reasoning will be in your internal <think> tags]
</thought>

<action>
```python
def {entry_point}(...):
    # fixed implementation
    pass
```
</action>
</instructions>"""
        else:
            output_instruction = """<instructions>
Provide your response following the ReAct pattern:

<thought>
[First, analyze test cases carefully - what inputs? what outputs? what patterns? Then identify the bug and plan your fix]
</thought>

<action>
```python
def {entry_point}(...):
    # fixed implementation
    pass
```
</action>
</instructions>"""
        
        prompt = f"""<system>
You are an expert Python debugger using the ReAct approach: Reasoning (think deeply about the bug) + Acting (propose a fix) + Observing (test and learn from results). {attempts_message}.
</system>

<critical_instructions>
Before proposing any fix, you MUST carefully analyze the test cases:
1. What are the exact inputs being tested?
2. What are the expected outputs?
3. Manually calculate: work through the FIRST test case by hand to understand the logic
4. What patterns or rules do you see in the input-output relationships?
5. Do edge cases reveal special requirements (e.g., only uppercase? only even indices? only certain conditions)?

Study the examples below to see how to analyze tests properly.
</critical_instructions>

{FEW_SHOT_EXAMPLES}

<task>
<buggy_code>
{state['buggy_solution']}
</buggy_code>

<tests>
{state['test']}
</tests>

<note>All tests must pass. Keep function name as {state['entry_point']}, make minimal necessary changes.</note>
</task>

{context}

{output_instruction.format(entry_point=state['entry_point'])}{thinking_directive}"""
        
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
                print(f"Observation: TESTS PASSED!\n")
            else:
                print(f"Observation: TESTS FAILED!\n{message}\n")
        
        # +++ If tests pass, this is our fixed solution
        # +++ If all attempts exhausted, return last attempt
        return {
            "test_passed": passed,
            "test_result": message,
            "fixed_solution": state['last_fix_attempt']
        }
    
    def route_after_test(state: AgentState) -> Literal["analyze", "end"]:
        """Route after Observation: continue ReAct loop or end."""
        # +++ Observation shows success -- we're done!
        if state['test_passed']:
            return "end"
        # +++ Max ReAct iterations reached - return best attempt
        elif state['iteration'] >= MAX_ITERATIONS:
            return "end"
        # +++ Continue ReAct cycle: Observation -> Thought + Action
        else:
            return "analyze"
    
    # Build graph - Pure ReAct: Thought+Action -> Observation -> loop
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