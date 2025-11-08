"""
LangGraph-based agent for fixing buggy Python code using Ollama with structured output and thinking mode.
"""

import asyncio
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from models import ProblemInput

try:
    from langchain_ollama import ChatOllama  # type: ignore
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except ImportError:
        ChatOllama = None  # type: ignore


# Model configuration
MODEL_NAME = 'hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL'
OLLAMA_HOST = 'http://localhost:11434'

# Debug flag
DEBUG = True


# Pydantic model for structured output
class CodeFixOutput(BaseModel):
    """Structured output from the code fixing agent."""
    
    fixed_code: str = Field(..., description="The corrected Python code")
    explanation: str = Field(
        default="",
        description="Brief explanation of what was fixed (optional)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence level in the fix"
    )


# LangGraph State
class AgentState(TypedDict):
    """State for the LangGraph agent."""
    
    problem: ProblemInput
    messages: list
    fixed_code: str
    explanation: str
    confidence: str
    streaming_output: str


def create_fix_prompt(problem: ProblemInput) -> str:
    """
    Create a prompt asking the LLM to fix the buggy code.
    
    Args:
        problem: The problem with buggy code to fix
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a Python debugging expert. Fix the buggy code below.

PROBLEM DESCRIPTION:
{problem.prompt}

BUGGY CODE:
```python
{problem.buggy_solution}
```

TASK:
1. Identify the bug in the code
2. Fix the bug
3. Return ONLY the corrected Python code without markdown code blocks or explanations

Provide the fixed code:"""
    
    return prompt


def analyze_problem(state: AgentState) -> AgentState:
    """
    Node 1: Analyze the problem and prepare messages.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with messages
    """
    problem = state["problem"]
    
    system_message = SystemMessage(
        content="You are a Python debugging expert. You fix buggy code precisely and return only the corrected code."
    )
    
    user_message = HumanMessage(content=create_fix_prompt(problem))
    
    state["messages"] = [system_message, user_message]
    state["streaming_output"] = ""
    
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"🔍 ANALYZING PROBLEM: {problem.task_id}")
        print(f"Bug type: {problem.bug_type}")
        print(f"{'='*80}\n")
        print("📝 BUGGY CODE:")
        print(problem.buggy_solution)
        print(f"\n{'-'*80}\n")
    
    return state


async def generate_fix_async(state: AgentState) -> AgentState:
    """
    Node 2: Call LLM to generate the fix (async with streaming).
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with fixed code
    """
    if DEBUG:
        print("🤖 MODEL OUTPUT (streaming):\n")
    
    full_response = ""
    use_langchain = ChatOllama is not None
    
    # Use ChatOllama if available, otherwise use ollama AsyncClient directly
    if use_langchain:
        try:
            # Initialize the LLM with LangChain
            # Settings optimized for Qwen3 with thinking and structured output modes
            # Reference: https://docs.unsloth.ai/models/qwen3-how-to-run-and-fine-tune
            llm = ChatOllama(
                model=MODEL_NAME,
                base_url=OLLAMA_HOST,
                temperature=0.6,  # Thinking mode: 0.6, Structured output: 0.7
                top_p=0.95,       # Thinking mode: 0.95, Structured output: 0.8
                top_k=20,         # Recommended for both modes
                mirostat=0,       # Disable mirostat (use standard sampling)
                num_predict=-1,   # No limit on output length
            )
            
            # Stream the response
            async for chunk in llm.astream(state["messages"]):
                content = chunk.content
                full_response += content
                state["streaming_output"] += content
                
                if DEBUG:
                    print(content, end="", flush=True)
        except Exception as e:
            if DEBUG:
                print(f"\nLangChain error: {e}, falling back to direct ollama client\n")
            # Fall back to direct ollama client
            use_langchain = False
            full_response = ""  # Reset response
    
    # Fallback: use ollama AsyncClient directly
    if not use_langchain:
        from ollama import AsyncClient
        
        client = AsyncClient(host=OLLAMA_HOST)
        
        # Convert messages to ollama format
        ollama_messages = []
        for msg in state["messages"]:
            if isinstance(msg, SystemMessage):
                ollama_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                ollama_messages.append({"role": "user", "content": msg.content})
        
        # Stream with ollama client (using Qwen3 optimized settings)
        async for part in await client.chat(
            model=MODEL_NAME, 
            messages=ollama_messages, 
            stream=True,
            options={
                'temperature': 0.6,
                'top_p': 0.95,
                'top_k': 20,
                'mirostat': 0,
                'num_predict': -1,
            }
        ):
            content = part['message']['content']
            full_response += content
            state["streaming_output"] += content
            
            if DEBUG:
                print(content, end="", flush=True)
    
    if DEBUG:
        print(f"\n\n{'-'*80}\n")
    
    # Clean up the response
    fixed_code = clean_code_response(full_response)
    
    state["fixed_code"] = fixed_code
    state["explanation"] = "Fixed by LLM"
    state["confidence"] = "medium"
    
    return state


def generate_fix(state: AgentState) -> AgentState:
    """
    Synchronous wrapper for generate_fix_async.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with fixed code
    """
    return asyncio.run(generate_fix_async(state))


def validate_output(state: AgentState) -> AgentState:
    """
    Node 3: Validate the output structure.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state (validated)
    """
    # Create structured output using Pydantic model
    try:
        structured_output = CodeFixOutput(
            fixed_code=state["fixed_code"],
            explanation=state["explanation"],
            confidence=state["confidence"]
        )
        
        # Update state with validated output
        state["fixed_code"] = structured_output.fixed_code
        state["explanation"] = structured_output.explanation
        state["confidence"] = structured_output.confidence
        
        if DEBUG:
            print("✅ OUTPUT VALIDATED")
            print(f"   Confidence: {structured_output.confidence}")
            print(f"   Code length: {len(structured_output.fixed_code)} chars")
            print(f"\n{'='*80}\n")
        
    except Exception as e:
        if DEBUG:
            print(f"⚠️  VALIDATION ERROR: {e}")
            print(f"   Using unvalidated output")
            print(f"\n{'='*80}\n")
    
    return state


def clean_code_response(response: str) -> str:
    """
    Clean the LLM response to extract only the code.
    
    Removes markdown code blocks, extra explanations, etc.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Cleaned code
    """
    import re
    
    # Remove markdown code blocks if present
    code_block_pattern = r'```(?:python)?\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code block, return the response as-is (stripped)
    return response.strip()


# Build the LangGraph
def build_agent_graph() -> StateGraph:
    """
    Build the LangGraph workflow for code fixing.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_problem)
    workflow.add_node("generate", generate_fix)
    workflow.add_node("validate", validate_output)
    
    # Define edges
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", END)
    
    # Compile the graph
    return workflow.compile()


# Create the agent graph (singleton)
agent_graph = build_agent_graph()


def fix_code(problem: ProblemInput, debug: bool = DEBUG) -> str:
    """
    Main entry point: Fix buggy code using LangGraph agent.
    
    Args:
        problem: The problem input containing buggy code
        debug: Whether to enable debug output (default: True)
        
    Returns:
        Fixed code as a string
    """
    global DEBUG
    DEBUG = debug
    
    # Initialize state
    initial_state: AgentState = {
        "problem": problem,
        "messages": [],
        "fixed_code": "",
        "explanation": "",
        "confidence": "medium",
        "streaming_output": ""
    }
    
    # Run the agent graph
    final_state = agent_graph.invoke(initial_state)
    
    return final_state["fixed_code"]
