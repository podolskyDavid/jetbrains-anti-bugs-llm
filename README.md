# Jetbrains LLM Benbchmarking project

## Task Description

As a test task, we invite you to implement an LLM-based AI agent that fixes buggy Python code and evaluate its quality using the Python subset of HumanEvalFix.

### Agent:

  * You can use any agentic framework (we suggest LangGraph).
  * You can implement any agentic scaffold (we suggest a ReAct-style agent that has access to a code interpreter tool–remember to sandbox execution of LLM-generated code).
  * You can use any LLM (we suggest a small open-source model that you could serve with limited computational resources like, e.g., Qwen3-0.6B).

### Evaluation:

* Your solution should include code for running the agent and obtaining benchmark scores. Provide instructions for running the code and share the results you got.
* Ideally, use the pass@1 metric from the paper. You are free to use the original implementation from the authors or any other existing implementation.
* If you find that the full Python subset takes too much time and/or resources, use a representative subsample.

## Implementation

### Stack

* LLM that will be tested: https://huggingface.co/unsloth/Qwen3-1.7B-GGUF. We will use this Ollama link: `ollama run hf.co/unsloth/Qwen3-4B-GGUF:UD-Q4_K_XL`
* python 3.11
* uv package manager/installer
* docker containers + docker compose to have a reproducable environment
* ollama to run the models
* Lang Graph for agents implementation
* Let's follow ReAct style + structured output + thinking mode of qwen3 models
* pydantic v2 (not v1!!!) for models
* no gpu!!! this is cpu-based implementation

## Setup & Running

### Using Docker (Universal, CPU-only)

```bash
docker-compose up -d
python src/testing.py
python src/run_benchmark.py
```

### Alternative: Native Ollama (Faster on Mac M-series)

For ~10-50x faster inference on Apple Silicon Macs:

```bash
brew install ollama
ollama serve &
ollama pull hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL
python src/testing.py
```

## Running the Benchmark

The benchmark script supports multiple modes:

### Default Mode (5 problems)
```bash
python src/run_benchmark.py
```

### Full Dataset (all 164 problems)
```bash
python src/run_benchmark.py --full
```

### Random Subset (percentage-based sampling)
```bash
# Run 33% of problems (~54 problems)
python src/run_benchmark.py --partial 33

# Run 50% of problems (~82 problems)
python src/run_benchmark.py --partial 50

# Run 10% of problems (~16 problems)
python src/run_benchmark.py --partial 10
```

### Help
```bash
python src/run_benchmark.py --help
```

**Note:** The `--partial` option randomly samples problems from the dataset, so results may vary between runs with the same percentage.