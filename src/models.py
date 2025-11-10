from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class AgentInput(BaseModel):
    """Input given to the agent (excludes solution and bug metadata)."""
    
    task_id: str
    # prompt: str  # not used in this implementation
    buggy_solution: str
    test: str  # supports a docs-only variant
    entry_point: Optional[str] = Field(None, description="Function name to fix")

    model_config = ConfigDict(extra="forbid")

class ProblemData(BaseModel):
    """Complete problem data from HumanEvalFix dataset."""
    
    # Core fields
    task_id: str
    prompt: str
    buggy_solution: str
    canonical_solution: str
    test: str
    entry_point: str
    
    # Required for test execution
    import_: Optional[str] = Field(None, alias="import")
    test_setup: Optional[str] = None
    
    # Optional metadata
    declaration: Optional[str] = None
    bug_type: Optional[str] = None
    failure_symptoms: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    instruction: Optional[str] = None
    example_test: Optional[str] = None
    
    model_config = ConfigDict(populate_by_name=True)

    def to_agent_input(self) -> AgentInput:
        return AgentInput(
            task_id=self.task_id,
            buggy_solution=self.buggy_solution,
            test=self.test
        )

class ProblemOutput(BaseModel):
    """Output from the agent for a single problem."""
    
    task_id: str
    fixed_solution: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


class BenchmarkResult(BaseModel):
    """Overall benchmark results."""
    
    total_problems: int = 0
    passed_problems: int = 0
    failed_problems: int = 0
    pass_at_1: float = 0.0
    results: list[ProblemOutput] = Field(default_factory=list)
    
    def add_result(self, result: ProblemOutput):
        """Add a problem result and update statistics."""
        self.results.append(result)
        if result.passed:
            self.passed_problems += 1
        else:
            self.failed_problems += 1
        self.total_problems = len(self.results)
        self.pass_at_1 = self.passed_problems / self.total_problems if self.total_problems > 0 else 0.0