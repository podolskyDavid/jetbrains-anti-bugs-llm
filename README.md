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

* LLM that will be tested: https://huggingface.co/unsloth/Qwen3-1.7B-GGUF. We will use this Ollama link: `ollama run hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q8_K_XL`
* python 3.11
* uv package manager/installer
* docker containers + docker compose to have a reproducable environment
* ollama to run the models
* Lang Graph for agents implementation
* Let's follow ReAct style + structured output + thinking mode of qwen3 models
* pydantic v2 (not v1!!!) for models
* no gpu!!! this is cpu-based implementation

### HumanEvalFix Bench Setup

**Dataset Structure:**
HumanEvalFix contains 164 Python programming problems derived from the original HumanEval benchmark. Each problem consists of a manually introduced bug in a canonical solution.

**Benchmark Variants:**
HumanEvalFix defines two evaluation protocols:
1. **HumanEvalFixTests** (this implementation): Agent receives buggy code and unit tests
2. **HumanEvalFixDocs**: Agent receives buggy code and natural language docstring only

**Agent Input (What the model sees):**
- `buggy_solution`: The incorrect Python function that needs to be fixed
- `test`: Unit test cases that must pass after fixing

This is the minimal standard input. The agent does NOT receive the problem description, docstring, or any hints about the bug type.

**Hidden from Agent (Used only for execution and evaluation):**
- `canonical_solution`: The correct implementation
- `bug_type`: Classification of the bug (missing logic, operator misuse, variable misuse, value misuse, function misuse, excess logic)
- `failure_symptoms`: Expected failure mode (incorrect output, infinite loop, stackoverflow)
- `prompt`, `docstring`, `example_test`: Problem descriptions (excluded in Tests variant)
- `entry_point`, `import`, `test_setup`: Metadata required for test execution

**Bug Categories:**
The dataset includes six bug types: missing logic, excess logic, operator misuse, variable misuse, value misuse, and function misuse. Bugs are subtle single-line changes to canonical solutions.

**Evaluation Protocol:**
1. Agent receives only buggy code and tests
2. Agent outputs fixed code
3. Fixed code is executed in sandboxed Docker container with 30-second timeout
4. Success measured using pass@1 metric: percentage of problems where the first generated solution passes all tests
5. Tests are executed against the complete original HumanEval test suite

**Why Tests-Only Input:**
This strict input format ensures comparability with published HumanEvalFix benchmarks. Providing both tests and docstrings creates a custom hybrid variant with incomparable results.

## Setup & Running (please run in non-thinking mode)

### Option 1: Docker Compose (Fully Isolated and works out of the box, but might be slower on apple m-chips)

```bash
# Option A: Quick run with auto-cleanup (recommended)
./run_benchmark.sh  # Runs benchmark and auto-stops all containers when done

# Option B: Manual control
# Start services (runs 10% of problems with debug output by default)
docker-compose up --build

# View logs in real-time (in another terminal)
docker-compose logs -f benchmark

# Or run different problem sets
docker-compose run --rm benchmark python src/run_benchmark.py --partial 33 --debug
docker-compose run --rm benchmark python src/run_benchmark.py --full --debug

# Run without debug output for faster execution
docker-compose run --rm benchmark python src/run_benchmark.py --partial 10

# Stop services
docker-compose down
```

**Results Location:**
Results are automatically saved to `results/benchmark_results.json` on your local machine (not inside the container). The `./results` directory is mounted as a volume, so all benchmark results persist on your computer even after containers are stopped/removed.

```bash
# View results
cat results/benchmark_results.json

# Or pretty-print with jq
cat results/benchmark_results.json | jq '.'
```

**Configuration Options:**

1. **Thinking Mode** - Edit `docker-compose.yml` to enable/disable:
   ```yaml
   environment:
     - ENABLE_THINKING=true   # Slower but potentially more accurate
     - ENABLE_THINKING=false  # Faster inference (default)
   ```
   Or use CLI flag: `--thinking`

2. **Debug Output** - Shows detailed LLM responses and test results:
   - Default command includes `--debug` flag
   - Remove it from `docker-compose.yml` command line for quieter output
   - Or pass `--debug` when using `docker-compose run`

3. **Problem Selection**:
   - Edit the `command:` line in `docker-compose.yml` to change `--partial 10` 
   - Or override: `docker-compose run --rm benchmark python src/run_benchmark.py --partial 50`

**Note:** First run downloads ~2GB model (takes a few minutes).

### Option 2: Local Development (Requires Ollama)

For development or when you have Ollama running locally:

```bash
# Install dependencies
uv pip install -r pyproject.toml

# Install Ollama
brew install ollama

# Start Ollama (if not already running)
ollama serve &
ollama pull hf.co/unsloth/Qwen3-1.7B-GGUF:UD-Q8_K_XL

# Run benchmarks
python src/run_benchmark.py                           # Default: 5 problems
python src/run_benchmark.py --partial 33              # 33% of problems (~54)
python src/run_benchmark.py --full                    # All 164 problems

# With debug output (shows LLM responses and detailed test results)
python src/run_benchmark.py --partial 10 --debug

# NOT RECOMMENDED!!! Enable qwen3 thinking mode
python src/run_benchmark.py --partial 10 --thinking --debug

# NOT RECOMMENDED!!! Set thinking mode via environment variable
export ENABLE_THINKING=true
python src/run_benchmark.py --partial 10 --debug

# View results
cat results/benchmark_results.json
```

**Results Location:**
All benchmark results are saved to `results/benchmark_results.json` in your project directory.

**Notes:** 
- The `--partial` option randomly samples problems from the dataset, so results may vary between runs with the same percentage.
- Use `--debug` flag to see detailed output including LLM streaming responses and test execution details.