import os
import re
from models import AgentInput
from typing import Literal, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Configuration
MODEL_NAME = os.getenv('OLLAMA_MODEL', 'hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q8_K_XL')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

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
    """State for the LangGraph agent."""
    buggy_solution: str
    test: str
    fixed_solution: str


def extract_code(text: str) -> str:
    """Extract Python code from LLM response, handling various formats."""
    # Try to find code in markdown blocks
    code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no markdown blocks, return the whole response stripped
    return text.strip()


def create_fix_code_graph(enable_thinking: bool, verbose: bool = False) -> StateGraph:
    """Create a simple LangGraph for fixing buggy code."""
    
    # Initialize LLM
    llm_params = get_llm_params(enable_thinking)
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_HOST,
        **llm_params
    )
    
    def fix_code_node(state: AgentState) -> AgentState:
        """Node that calls LLM to fix the buggy code."""
        thinking_directive = get_thinking_directive(enable_thinking)
        
        prompt = f"""You are an expert Python debugger working on fixing buggy production Python code. Your job is to apply the *smallest possible textual change* that makes all provided tests pass.

CONTEXT
-------
You will be given buggy Python code and the tests that must pass.

BUGGY CODE:
{state['buggy_solution']}

TEST CASES (must all pass):
{state['test']}

OBJECTIVE
---------
Produce a *minimal* fix that makes **all** tests pass while preserving the original API and structure.

HARD RULES (follow all)
-----------------------
1) Preserve the function **name**, **parameters**, and **return type** exactly as in the buggy code.
2) Do **not** add or remove top-level code: no new imports, no print/logging/asserts, no new helper functions, no I/O, no globals.
3) Make the **fewest token-level edits** required (flip a comparator, add `abs`, fix index/variable, off-by-one, integer vs float division, base case, fencepost).
4) If multiple fixes work, prefer the **smallest textual edit**. Priority: (1) pass tests, (2) match docstring/spec if present, (3) preserve style, (4) minimize diff.
5) Deterministic output: no comments, docstrings, explanations, extra whitespace churn, or refactors.

OUTPUT SPEC (very strict)
-------------------------
- You must output **exactly one** fenced Python code block containing the **complete fixed function**, surrounded by these sentinels:
__BEGIN_FIXED_CODE__
```python
# fixed function here
```
__END_FIXED_CODE__
- Do **not** include any other backtick fences anywhere.
- No text before or after the sentinels.

Remember: smallest edit that makes all tests pass. Now produce the result.{thinking_directive}"""

        
        if verbose:
            print("+++ LLM RESPONSE (streaming) +++\n")
        
        # Stream the response and accumulate chunks
        full_response = ""
        for chunk in llm.stream(prompt):
            if hasattr(chunk, 'content'):
                content = chunk.content
                full_response += content
                if verbose:
                    print(content, end="", flush=True)
        
        if verbose:
            print("\n")
        
        fixed_code = extract_code(full_response)
        
        return {"fixed_solution": fixed_code}
    
    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("fix_code", fix_code_node)
    graph.add_edge(START, "fix_code")
    graph.add_edge("fix_code", END)
    
    return graph.compile()


def fix_code_agent(agent_input: AgentInput, enable_thinking: bool, verbose: bool = False, bug_type: str = None) -> str:
    """
    Main entry point: Fix buggy python code using LangGraph agent.
    
    Args:
        agent_input: The agent input
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

    # Create and run the graph
    graph = create_fix_code_graph(enable_thinking, verbose=verbose)
    
    initial_state: AgentState = {
        "buggy_solution": agent_input.buggy_solution,
        "test": agent_input.test,
        "fixed_solution": ""
    }
    
    result = graph.invoke(initial_state)
    
    return result["fixed_solution"]