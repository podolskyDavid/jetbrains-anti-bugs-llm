"""
Pydantic models for HumanEvalFix benchmarking.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ProblemInput(BaseModel):
    """Input data for a single problem from HumanEvalFix dataset."""
    
    task_id: str = Field(..., description="Unique identifier for the problem")
    prompt: str = Field(..., description="Problem description/docstring")
    buggy_solution: str = Field(..., description="The buggy code that needs to be fixed")
    canonical_solution: str = Field(..., description="Reference correct solution")
    test: str = Field(..., description="Test cases to verify the solution")
    entry_point: str = Field(..., description="Name of the function to test")
    declaration: Optional[str] = Field(None, description="Function declaration/signature")
    bug_type: Optional[str] = Field(None, description="Type/category of the bug")
    example_test: Optional[str] = Field(None, description="Example test case")
    
    class Config:
        extra = "allow"  # Allow additional fields from dataset


class ProblemOutput(BaseModel):
    """Output from the agent for a single problem."""
    
    task_id: str = Field(..., description="Unique identifier for the problem")
    fixed_solution: str = Field(..., description="The agent's fixed code")
    passed: bool = Field(..., description="Whether all tests passed")
    error_message: Optional[str] = Field(None, description="Error message if tests failed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute tests (seconds)")
    
    class Config:
        extra = "allow"


class BenchmarkResult(BaseModel):
    """Overall benchmark results across all problems."""
    
    total_problems: int = Field(..., description="Total number of problems evaluated")
    passed_problems: int = Field(..., description="Number of problems that passed all tests")
    failed_problems: int = Field(..., description="Number of problems that failed")
    pass_at_1: float = Field(..., description="pass@1 metric (passed/total)")
    results: list[ProblemOutput] = Field(default_factory=list, description="Individual problem results")
    
    def add_result(self, result: ProblemOutput):
        """Add a problem result and update statistics."""
        self.results.append(result)
        if result.passed:
            self.passed_problems += 1
        else:
            self.failed_problems += 1
        self.total_problems = len(self.results)
        self.pass_at_1 = self.passed_problems / self.total_problems if self.total_problems > 0 else 0.0

