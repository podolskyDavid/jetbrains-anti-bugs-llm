As a test task, we invite you to implement an LLM-based AI agent that fixes buggy Python code and evaluate its quality using the Python subset of HumanEvalFix.

## Agent:

  * You can use any agentic framework (we suggest LangGraph).
  * You can implement any agentic scaffold (we suggest a ReAct-style agent that has access to a code interpreter tool–remember to sandbox execution of LLM-generated code).
  * You can use any LLM (we suggest a small open-source model that you could serve with limited computational resources like, e.g., Qwen3-0.6B).

## Evaluation:

* Your solution should include code for running the agent and obtaining benchmark scores. Provide instructions for running the code and share the results you got.
* Ideally, use the pass@1 metric from the paper. You are free to use the original implementation from the authors or any other existing implementation.
* If you find that the full Python subset takes too much time and/or resources, use a representative subsample.

