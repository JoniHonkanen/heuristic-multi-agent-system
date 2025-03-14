from enum import Enum
from typing import List, TypedDict, Optional
from pydantic import BaseModel, Field, Extra, validator


# Define an Enum for the proceed field
class ProceedOption(str, Enum):
    CONTINUE = "continue"
    CANCEL = "cancel"
    NEW = "new"
    DONE = "done"
    FIX = "fix"


class SolutionMethod(str, Enum):
    HEURISTIC = "heuristic"
    OPTIMIZATION = "optimization"


# Schema for whole code project
class Purpose(BaseModel):
    user_summary: str = Field(
        description="A summarized statement of what the user wants to achieve."
    )
    problem_type: str = Field(
        description="The type of problem that is being solved, e.g., logistics optimization, resource allocation, etc."
    )
    optimization_focus: str = Field(
        description="A description of how the solution should be optimized, including key objectives such as minimizing costs or maximizing efficiency, and if applicable, specifying preferred heuristic methods like Nearest Neighbor for routing or Genetic Algorithms for cutting stock problems."
    )
    chatbot_response: str = Field(
        description="The chatbot's response to the user, explaining what is the problem, what will be done to solve the problem and why this approach is being taken."
    )
    goal: str = Field(
        description="The core objective of the task, summarizing the user's real goal and the purpose of solving the task. The goal should clearly define what needs to be achieved and include relevant key performance indicators (KPIs) or success criteria. These may include metrics such as cost minimization, efficiency improvement, resource utilization... "
    )
    resource_requirements: str = Field(
        description="Specific requirements or allocations for the resource, such as how much is required for each task, order, or destination. This should include all the important details needed to solve the problem effectively."
    )
    solution_method: SolutionMethod = Field(
        description="Determines whether the solution should use heuristic optimization or code generation."
    )


class Code(BaseModel):
    python_code: str = Field(
        description="Python code generated by the AI agents to solve the user's problem."
    )
    requirements: Optional[str] = Field(
        default="No requirements provided",  # Default value in case it's missing
        description="A list of requirements or dependencies needed to run the generated Python code, such as those used in a requirements.txt file.",
    )
    resources: Optional[str] = Field(
        default="No additional resources provided",  # Default value in case it's missing
        description="If no resources added, use default answer. Any additional requirements or files, such as data sheets (Excel files) or other resources, that are not included in the main requirements list but are necessary for the program.",
    )
    used_heuristic: Optional[str] = Field(
        default=None,
        description="The heuristic method used to generate the code, if applicable. The agent should recognize and add a known optimization method if relevant."
    )


class CodeFix(BaseModel):
    fixed_python_code: str = Field(
        description="Python code generated by the AI agents after fixing the user's problem."
    )
    requirements: Optional[str] = Field(
        default="No requirements provided",
        description="List of requirements or dependencies needed to run the fixed Python code, adjusted if necessary.",
    )
    requirements_changed: bool = Field(
        default=False,
        description="Indicates if requirements were changed to resolve the issue.",
    )
    fix_description: str = Field(
        description="Description of what was fixed, providing context for the changes made to the code."
    )
    original_error: str = Field(
        description="Summary of docker logs (provided error) what needed to be fixed and why."
    )


class DockerFiles(BaseModel):
    dockerfile: str = Field(
        description="The Dockerfile that defines the Docker environment for running the generated Python code."
    )
    compose_file: str = Field(
        description="The docker-compose.yaml file that specifies the services and configurations for the Docker environment."
    )


class OutputOfCode(BaseModel):
    answer: str = Field(
        description="The extracted numerical result or key output from the calculation (e.g., total distance, optimized cost, material usage). If a unit is provided in the output, it should be included in the relevant field, but no assumptions should be made about units if they are not explicitly stated."
    )
    answer_description: str = Field(
        description="A detailed explanation of the numerical result, providing insights into its meaning, relevance, and any constraints or conditions affecting the outcome."
    )
    improvement: str = Field(
        description="Potential optimizations, refinements, or alternative approaches that could improve the solution, reduce resource usage, or enhance efficiency."
    )
    objective_value: Optional[float] = Field(
        description="The final objective value computed by the optimization algorithm (e.g., minimized cost, total distance, or another key performance metric). Units should only be included if explicitly mentioned in the output.",
        default=None,
    )
    explanation: str = Field(
        description="A step-by-step breakdown of how the algorithm reached the final result, including key decisions, constraints, and assumptions made during the process."
    )
    is_goal_achieved: str = Field(
        description="Boolean value indicating whether the goal was achieved, followed by a detailed explanation of why the goal was met or not, referencing the extracted results."
    )
    code: Optional[str] = Field(
        default=None,
        description="The generated Python code used to produce the result. This will be added later and should remain unchanged until explicitly modified."
    )


class FinalReport(BaseModel):
    index_of_optimization: int = Field(
        description="From all the optimizations, which one is the best. Use given index to find the best optimization."
    )
    reason: str = Field(description="Why this optimization is the best.")


class AgentState(TypedDict):
    userInput: str  # Original user input
    iterations: int  # Number of iterations (Not used yet)
    promptFiles: List[str]  # Given files whats been uploaded
    messages: List[str]
    purpose: Purpose  # What user want to achieve
    proceed: ProceedOption  # Enum
    code: Code  # Python code and requirements
    dockerFiles: DockerFiles  # DockerFile and compose.yaml
    docker_output: str  # What running code in docker container outputs
    result: OutputOfCode  # Results of the code execution - answer, explanation, etc.
    results: List[OutputOfCode]
    solution_method: SolutionMethod
