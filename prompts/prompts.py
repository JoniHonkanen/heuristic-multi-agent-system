from langchain_core.prompts import ChatPromptTemplate

TASK_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to carefully analyze the user's input and any provided data to understand the problem they are trying to solve. The task could involve logistics process optimization (e.g., finding the shortest or fastest routes), the cutting stock problem, or other similar challenges.

    **Step 1: Understanding the User's Objective**
    - Identify the core purpose of the task and what the user ultimately wants to achieve.
    - Summarize the user's real objective in your own words, based on the task description and any available data (Python code, Excel files, or no data at all).
    - If no data is provided, identify what might be missing to solve the task.

    **User task description:**
    {user_input}

    **Provided data (Python code, Excel files, VRP file, or may be empty):**
    {data}

    **Step 2: Target Value Handling**
    - **Determine if the user has specified a target value for optimization** (e.g., a minimum distance, maximum efficiency, or cost constraint).
    - If a target value is given, analyze whether it is:
      - **Feasible** based on the problem constraints.
      - **Challenging but possible** with improved optimization techniques.
      - **Unrealistic**, requiring adjustments to constraints or expectations.
    - If no target is explicitly given, infer what an optimal outcome might look like based on the problem description.

    **Step 3: Data Quality & Missing Information**
    - Analyze the available data and determine if it is **sufficient** for solving the optimization problem.
    - Identify any missing **inputs, constraints, or parameters** that are required for accurate results.
    - If the data is incomplete, suggest **which additional inputs the user should provide**.

    **Step 4: Problem Classification**
    - Determine the **problem type** (e.g., logistics optimization, scheduling, vehicle routing, resource allocation).
    - Define the **optimization focus**: What key aspects of the solution should be improved? (e.g., minimizing cost, reducing travel distance, maximizing efficiency).
    - Consider whether the problem has a **strict mathematical formulation** or whether heuristic/metaheuristic methods are required.

    **Step 5: Algorithm Selection & Justification**
    - Based on the problem type, determine whether the solution should be:
      - **An exact optimization approach** (e.g., **PuLP for linear programming**).
      - **A heuristic or metaheuristic approach** (e.g., **A* search, simulated annealing, genetic algorithms, tabu search**).
    - **Justify why the chosen algorithm is appropriate** based on:
      - **Problem complexity** (e.g., NP-hard, large-scale)
      - **Computational efficiency needs** (e.g., real-time optimization)
      - **Solution quality vs. execution speed trade-off**
    - If the user has specified a **target value**, consider whether:
      - The current approach is capable of achieving it.
      - A different optimization technique should be used.

    **Step 6: Heuristic Evaluation & Further Refinements**
    - If a heuristic/metaheuristic method is used, evaluate whether **it is sufficient or if additional optimization steps are required**.
    - If using **nearest_neighbor** as an initial heuristic, specify how the solution should be **further optimized (e.g., `GUIDED_LOCAL_SEARCH`, genetic algorithm tuning)**.
    - If the target value appears difficult to reach, suggest:
      - Adjustments to constraints.
      - A more advanced optimization technique.

    **Step 7: Solution Type Evaluation**
    - Determine whether the generated solution should be:
      - **Exact and optimal** (e.g., using integer programming or branch-and-bound methods).
      - **Approximate and heuristic-based** (e.g., a metaheuristic search algorithm).
    - Explain the **trade-offs** between computational time and solution quality.
    - If the target value is given, determine whether the chosen method can achieve it efficiently.

    **Final Summary**
    - **What is the user trying to achieve with this task?** (Summarize the user's real objective)
    - **What is the purpose of solving this task?**
    - **Is there a defined target value, and is it achievable?**
    - **How can the solution be optimized to achieve the best result, focusing on quality and problem-solving?**
    - **Are there any missing inputs that should be collected before generating code?**
    - **Which heuristics or metaheuristics could be beneficial, and why?**
    - **Does the heuristic require further refinement or additional steps for better results?**
    - **Is the solution expected to be exact or approximate, and why?**
    """
)


CODE_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to generate **fully functional, optimized Python code** that solves the optimization problem described by the user. The solution must be based on **exact mathematical optimization methods**.

    Based on the user's input and the provided files (e.g., Python code, Excel sheets), generate Python code that implements **either linear programming (LP) or integer programming (ILP/MILP)** using PuLP. If PuLP is unsuitable, consider **OR-Tools constraint programming** as an alternative.

    **Choosing the Right Optimization Approach**
    - **PuLP is the preferred method** for solving **Linear Programming (LP), Integer Programming (ILP), and Mixed-Integer Linear Programming (MILP)** problems.
    - **Use PuLP whenever possible.** Only use OR-Tools if PuLP cannot handle the problem's structure.
    - **The optimization model must be well-defined**, including:
      - **Decision variables**
      - **Objective function** (minimization or maximization)
      - **Constraints** (e.g., resource limits, time windows, scheduling conditions)
    - **Avoid heuristic-based methods**—the approach must be mathematically sound and guarantee optimality.

    **Ensuring Correct Use of PuLP**
    - **Define decision variables properly**, ensuring that they represent the key components of the optimization problem.
    - **Specify constraints explicitly**—use `LpConstraint` to define inequalities.
    - **Use `LpVariable(..., cat=LpInteger)`** for integer constraints when required.
    - **Validate feasibility**: If the model is infeasible, adjust constraints accordingly.
    - **If an LP/MILP model cannot be built, OR-Tools may be used** as an alternative.

    **Summary of the user's input:**
    {user_summary}

    **Problem type identified in earlier analysis:**
    {problem_type}

    **Optimization focus:**
    {optimization_focus}

    **Provided data (Python code, Excel files, or may be empty):**
    {data}
    
    **Resource Requirements:**
    {resource_requirements}

    **Response Format**
    The output should be structured as a JSON object with the following fields:

    - **python_code**: The optimized, error-free Python code that solves the problem using PuLP (or OR-Tools if necessary).
    - **requirements**: A list of all required Python packages (e.g., `pandas`, `PuLP`, `ortools`). Do not include specific versions unless absolutely necessary.
    - **resources**: List any additional files, external datasets, or dependencies required for execution.
    - **optimization_analysis**: A brief explanation of why the chosen mathematical optimization method was used, comparing alternative methods.

    **Handling File Inputs & Data Processing**
    - If the provided files include **data sheets (Excel, CSV, etc.)**, ensure that:
      - **The correct file handling libraries (`pandas`, `openpyxl`) are included** in the requirements.
      - The code properly reads, processes, and integrates the data into the optimization model.
      - **The data format is preserved** to avoid errors (e.g., numerical precision issues, encoding conflicts).
    - **Avoid hardcoded paths**—ensure that input filenames are configurable.

    **Quality Assurance & Error Handling**
    - **Ensure the generated Python code is mathematically sound and produces optimal solutions.**
    - Validate that **all constraints and objective functions** are correctly formulated.
    - Before returning the final code, analyze it for **syntax errors, incorrect logic, or unbounded solutions**.
    - **Ensure all required dependencies are properly imported**.
    - Include **at least one test case using Python's `unittest`** to verify correctness.
    - If numerical results are involved, include **assertions that validate expected values**.
    - **Avoid brute-force or exhaustive search methods**—the approach must be mathematically optimized.

    **Example `requirements.txt` (flexible installation)**
    ```plaintext
    pandas
    PuLP
    numpy
    scipy
    openpyxl
    ortools
    ```

    - Avoid pinning specific versions unless required for compatibility.

    **Ensuring Correct Optimization**
    - If an error occurs during execution, analyze the failure, correct the issue, and return the fixed version.
    - If the generated solution is infeasible, refine constraints and try again.
    - **Clearly document assumptions and constraints** in the generated code.
    - If a better mathematical optimization method exists, reprocess and return the improved solution.

    **Final Requirement:** The generated Python code **must be structured, fully functional, and optimized for accuracy and efficiency**.
    """
)



CODE_PROMPT_NO_DATA = ChatPromptTemplate.from_template(
    """
    Your task is to generate Python code that solves the optimization problem described by the user, using either PuLP (for linear programming) or heuristic algorithms (e.g., genetic algorithms, simulated annealing) depending on the problem's complexity and requirements.

    Based on the user's input, generate Python code that implements either exact optimization (using PuLP) or heuristic algorithms where an approximate solution is more practical. The goal is to develop a solution that efficiently addresses the problem's constraints and objectives.

    The code must be fully functional and structured to solve the task optimally or near-optimally, depending on the method chosen. You should also define any necessary test parameters derived from the user input to validate the solution.

    Additionally, ensure that all packages specified in `requirements` include compatible version numbers to avoid conflicts. **The package versions must be compatible with each other to prevent dependency issues**. Research the compatibility requirements if needed and select versions that are widely compatible with standard dependencies (e.g., PuLP, pandas, openpyxl).

    In your response, generate the following fields:
    - **python_code**: The fully functional Python code that solves the user's problem.
    - **requirements**: A comprehensive list of all Python packages or dependencies (e.g., pandas, PuLP) needed to run the generated Python code. This will be used to create a requirements.txt file.

    Summary of the user's input:
    {user_summary}

    Problem type identified in earlier analysis:
    {problem_type}

    Optimization focus:
    {optimization_focus}
    
    Resource Requirements:
    {resource_requirements}
    
    **Note:** The **Resource Requirements** section is very important for the generated code. It details the key resources available (e.g., materials, vehicles, personnel) and specific requirements (e.g., quantities, sizes) that must be fulfilled to solve the problem. The generated code must prioritize these resource requirements when forming the solution, ensuring that all available resources are utilized efficiently and constraints are respected.

    The code **must accurately represent the problem** based on the user's input, ensuring that all key factors (e.g., materials, quantities, constraints) relevant to the user's task are considered.

    ### Key Points to Address:
    - What is the optimization problem, and what constraints or requirements need to be considered?
    - Should the solution use PuLP for exact optimization, or is a heuristic algorithm more appropriate for solving this problem?
    - How should the code structure reflect the optimization method to ensure clarity and efficiency?
    - Define parameters for testing, based on user input, to validate the optimization approach.
    - What packages or libraries are required for requirements.txt to run the generated Python code, including any necessary for file handling (e.g., pandas, openpyxl) if provided data includes Excel or other files? Ensure to handle potential encoding issues in file reading to avoid errors.
    - **Ensure the `requirements` section lists all necessary Python packages without specifying exact versions unless required by specific compatibility needs. This allows pip to install the latest compatible versions automatically.**

    Example of how `requirements.txt` could look:

    ```plaintext
    pandas
    PuLP
    openpyxl
    ```

    - This example avoids version pinning and allows pip to install the latest compatible versions of the required packages.

    If version pinning is needed for specific reasons (e.g., due to compatibility issues or reproducibility requirements), here is an example of how to specify versions:

    ```plaintext
    pandas==2.1.1
    PuLP==2.9.0
    openpyxl==3.1.5
    ```

    - **Make sure the code outputs the final result clearly (e.g., using print statements or by returning the result in a structured format like a table or numerical answer).**
    """
)

HEURISTIC_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to generate fully functional, error-free Python code that implements an efficient heuristic-based solution to solve the optimization problem described by the user. The code must execute without issues on Python 3.9+.

    Constraints:
    - Ensure the Python code is free of syntax errors, undefined variables, or missing imports.
    - Every function call must be defined properly before being used.
    - Do not include deprecated functions or libraries.
    - Check that all required packages exist in PyPI and are installable in Python 3.9+.
    - Use only well-maintained packages (e.g., pandas, numpy, networkx).
    - Avoid unnecessary dependencies—only include packages that are actually used in the code.
    - Do not use exact package versions (==), use flexible versions (>=latest_stable_version).
    - The generated code must run correctly without requiring modifications.

    Summary of the user's input:
    {user_summary}

    Problem type identified in earlier analysis:
    {problem_type}

    Optimization focus:
    {optimization_focus}

    Provided data (Python code, Excel files, `.vrp` files, or may be empty):
    {data}

    Resource Requirements:
    {resource_requirements}

    Choosing the best heuristic approach:
    - For routing problems (VRP, TSP, logistics optimization): Use Nearest Neighbor (NN) or Clarke-Wright Savings for quick solutions. For more refined results, apply Simulated Annealing (SA) or Tabu Search. For large-scale problems, use Genetic Algorithms (GA) or Ant Colony Optimization (ACO).
    - For cutting stock problems: Use First-Fit Decreasing (FFD) or Best-Fit Decreasing (BFD). For better results, apply Simulated Annealing (SA) or Genetic Algorithms (GA).
    - For scheduling and bin-packing problems: Use Greedy algorithms (Earliest Deadline First, Shortest Processing Time First). For complex constraints, use Tabu Search or Particle Swarm Optimization (PSO).
    - For large-scale combinatorial problems: If a well-known heuristic exists, use it (e.g., GRASP, Variable Neighborhood Search). If the problem is complex, try evolutionary algorithms (Genetic Algorithm, Differential Evolution).
    - For multi-objective optimization: Use NSGA-II (Non-Dominated Sorting Genetic Algorithm) to balance competing objectives.

    Handling VRP & Routing-Specific Files:
    - If the provided files include `.vrp` files (Vehicle Routing Problem data):
      - Ensure that all problem parameters (e.g., vehicle capacities, distance matrices) are correctly parsed.
      - Use `networkx` or custom parsing functions to extract graph-based representations.
      - Implement the best routing heuristic (e.g., Nearest Neighbor for quick solutions, Simulated Annealing for improved results).
      - Ensure constraints such as vehicle capacity and time windows are respected.
      - Optimize travel distance and minimize total route cost efficiently.

    Handling dependencies:
    - The `requirements` field must include only essential dependencies.
    - Ensure all packages are installable in Python 3.9+.
    - If a package version is unavailable, replace it with a valid alternative.
    - Use flexible dependency versions (>=latest_stable_version) instead of exact version pins (==).
    - Remove unnecessary dependencies.

    Example requirements.txt:
    pandas>=1.3
    numpy>=1.21
    networkx>=2.6
    deap>=1.3
    scipy>=1.7
    ortools>=9.2

    Ensuring correctness:
    - The Python code must be executable without modifications.
    - All functions must be properly defined before being called.
    - All required packages in requirements.txt must be installable.
    - No missing variables, functions, or logic errors.
    - The generated code must run at least one test case to confirm correctness.

    Response format:
    Return a JSON object with the following fields:
    - python_code: The fully functional, error-free Python code implementing the heuristic.
    - requirements: A list of required Python packages.
    - resources: List of additional required files or datasets.

    The generated Python code must be structured, fully functional, and optimized for efficiency and scalability.
    """
)


DOCKER_FILES_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to create the necessary Docker environment files to run the Python code generated for the optimization problem. This includes creating a Dockerfile and a compose.yaml file. The requirements.txt file (in the Generated requirements part) has defined the project's dependencies.

    Start by creating a Dockerfile that specifies the base image, environment setup, and commands needed to run the Python code. Ensure that the Dockerfile includes all the necessary configurations to execute the generated code successfully.

    Then, create a compose.yaml file that sets up the Docker container for the Python code. The compose.yaml should define the services and configurations needed to build and run the code.

    The generated Python code will always be saved as **generated.py**, so your Dockerfile and compose.yaml should be set up accordingly.

    For example:
    ```yaml
    services:
      app:
        build: .
        container_name: my-python-app
        command: python /app/generated.py  # Modify the command as per your project
    ```

    Generated Python code:
    {python_code}

    Generated requirements (requirements.txt):
    {requirements}
    
    Remember to include any additional resources or files that are required to run the Python code but are not listed in the main requirements.
    {resources}

    Key tasks:
    - Create a Dockerfile with the necessary configurations to run the generated Python code.
    - Create a compose.yaml file (Docker Compose V2) that defines the services to build and run the code.
    - Ensure the setup is ready for local testing by building and running the container.
    """
)

CODE_OUTPUT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to analyze the output generated by the Python code and parse the relevant information, such as numerical results or tables, to directly answer the user's question. This involves examining the output data, identifying any issues or discrepancies, and providing insights into the quality of the solution.

    **Original Question:**
    {user_summary}
    
    **Original Goal:**
    {original_goal}

    **Output of the Code:**
    {code_output}

**Instructions:**

- **Parse the Numerical Results or Tables:** 
  - Extract the relevant numerical answers or tables from the code output (e.g., how materials should be cut, total waste, objective value, etc.).
  - Present these results clearly and concisely.
- **Check against the Planned Steps:**
  - Verify that the final output reflects the steps outlined in "planned steps part".
  - Ensure that the results match the plan and that each step in the process has been executed correctly.
  - If any planned steps were skipped or not fully implemented, provide details.
- **Answer the Original Question:** 
  - Based on the parsed results, provide a clear and direct answer to the user's question.
- **Evaluate the Output Data:** 
  - Does the output data look correct and logically consistent?
  - Are there any issues, errors, or discrepancies present?
- **Assess the Solution:**
  - Has the original question or problem been effectively solved?
  - How well does the solution meet the requirements?
- **Summarize the Results:** 
  - Highlight key findings or outputs from the code (e.g., how the materials were allocated, total waste, etc.).
  - Provide any relevant metrics or outcomes.
- **Provide Insights and Recommendations:** 
  - Offer any observations about the quality or efficiency of the solution.

**Your response should be thorough, accurate, and suitable for use in further analysis or decision-making.**
    """
)

NEW_LOOP_CODE_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to generate a **fully optimized and improved version** of the Python code that solves the optimization problem described by the user. This new version must be based on previous iterations and must **explicitly enhance the previous solution by achieving a better objective value** (e.g., lower cost, shorter distance, higher efficiency).

    **Step 1: Analyze the Previous Solution**
    - Review the last used code and its results to determine if it **met the target goal**.
    - Identify why the previous solution was **not optimal**:
      - Did it fail to reach the given target?
      - Were there inefficiencies in the approach?
      - Could computational performance be improved?
    - If the previous solution met the goal, determine if **further improvements are still possible**.
    - Identify **which parts of the previous code were the main limitations** and require the most improvement.

    **Step 2: Generate an Improved Solution**
    - Produce a new version of the code that **strictly improves upon the last iteration**.
    - If the problem is a minimization problem (e.g., distance, cost), the new result **must be lower** than the previous one.
    - If the problem is a maximization problem (e.g., profit, efficiency), the new result **must be higher** than the previous one.
    - Avoid duplicating unnecessary parts of the previous code.
    - **Ensure the new code is computationally more efficient** (e.g., reduced time complexity, better memory handling).
    - If a different optimization technique can yield better results, **apply and justify the change**.
    - Clearly document all improvements with inline comments explaining why changes were made.

    **Step 3: Validate and Compare the Improvement**
    - **Run a comparison test** between the previous solution and the newly generated one.
    - If the new solution is **not strictly better**, retry optimization using a different approach.
    - Ensure that the new code is **computationally efficient** and does not introduce unnecessary complexity.
    - **If heuristics were used, determine whether a more advanced search method (e.g., simulated annealing, tabu search) can further enhance results**.
    - If the solution is **not consistently better**, introduce an iterative improvement mechanism.

    **Step 4: Output the Best Solution**
    Your response must include:

    - **python_code**: The fully optimized Python code that strictly improves the previous result.
    - **requirements**: A list of all Python dependencies (e.g., pandas, PuLP) needed to run the generated code.
    - **performance_comparison**: A summary of how the new solution compares to the previous one.
    - **objective_value**: The final computed value of the objective function in both the previous and new versions.
    - **test_cases**: A brief description of test cases used to verify that the new solution is indeed better.

    **Original user input:**
    {user_summary}

    **Original problem type:**
    {problem_type}

    **Optimization focus:**
    {optimization_focus}

    **Results from prior optimizations:**
    {previous_results}
    
    **Last used code:**
    {previous_code}

    **Provided data (Python code, Excel files, VRP file, or may be empty):**
    {data}
    
    **Resource Requirements:**
    {resource_requirements}

    **Final Requirements**
    - Ensure the generated code **is fully functional, free from syntax errors, and improves on the previous version**.
    - The new objective value **must be strictly better** than the previous one.
    - If the new solution is **not better**, retry with a different optimization method.
    - Ensure that the code outputs the **final computed objective value** clearly for easy comparison.
    - Automate testing: Generate test cases that verify the improvement.
    - Optimize runtime efficiency **without compromising solution quality**.
    """
)


NEW_LOOP_CODE_PROMPT_NO_DATA = ChatPromptTemplate.from_template(
    """
    Your task is to generate Python code that solves the optimization problem described by the user, using either PuLP (for linear programming) or heuristic algorithms (e.g., genetic algorithms, simulated annealing) depending on the problem's complexity and requirements.

    You should also take into account the results from previous optimization iterations to further improve the solution. Analyze the previously generated results (e.g., cutting patterns, waste minimization, resource utilization) and incorporate those insights into the next round of optimization. Ensure that the new solution is an improvement over the prior result and **does not duplicate any previous code**.

    Based on the user's input and the outcomes from the previous iteration(s), generate Python code that implements either exact optimization (using PuLP) or heuristic algorithms where an approximate solution is more practical. The goal is to develop a solution that efficiently addresses the problem's constraints and objectives, while **explicitly improving on the last used code**.

    The new code must:
    - Be fully functional and structured to solve the task optimally or near-optimally, depending on the method chosen.
    - Avoid duplicating any parts of the previous code and **explicitly comment on what improvements have been made** in terms of efficiency, structure, or the optimization goal (e.g., less waste, faster routes).
    - Define any necessary test parameters derived from the user input and previous results to validate the new solution.

    In your response, generate the following fields:
    - **python_code**: The fully functional Python code that solves the user's problem, improving on previous iterations and avoiding duplication of the last used code.
    - **requirements**: A comprehensive list of all Python packages or dependencies (e.g., pandas, PuLP) needed to run the generated Python code.

    **Original user input:**
    {user_summary}

    **Original problem type:**
    {problem_type}

    **Original optimization focus:**
    {optimization_focus}

    **Results from prior optimizations:**
    {previous_results}
    
    **Last used code:**
    {previous_code}

    The code **must accurately represent the problem** based on the user's input, ensuring that all key factors (e.g., materials, quantities, constraints) relevant to the user's task are considered. The previous optimization results should also guide the current solution in refining the approach.

    Resource Requirements:
    {resource_requirements}
    
    **Note:** The **Resource Requirements** section is very important for the generated code. It details the key resources available (e.g., materials, vehicles, personnel) and specific requirements (e.g., quantities, sizes) that must be fulfilled to solve the problem. The generated code must prioritize these resource requirements when forming the solution, ensuring that all available resources are utilized efficiently and constraints are respected.

    Key points to address:
    - What is the optimization problem, and what constraints or requirements need to be considered?
    - Should the solution use PuLP for exact optimization, or is a heuristic algorithm more appropriate for solving this problem?
    - **How does the new code differ from the last used code?** What improvements have been made to avoid duplication and enhance the previous solution?
    - Define parameters for testing, based on user input and the results of previous optimizations, to validate the new solution.
    - What packages or libraries are required for requirements.txt to run the generated Python code?
    - **Make sure the code outputs the final result clearly (e.g., using print statements or by returning the result in a structured format like a table or numerical answer), with improvements from the previous iteration and no duplication.**
    - Ensure the generated Python code is **free from syntax errors**. All functions should be properly defined, and indentation should follow Python's strict indentation rules. Ensure all variables, function definitions, and imports are correctly structured.
    """
)

NEW_LOOP_HEURISTIC_PROMPT = ChatPromptTemplate.from_template(
    """
    Your task is to generate a **strictly improved version** of the heuristic-based Python code that solves the optimization problem described by the user. This new version must be based on previous iterations and must **explicitly enhance the previous heuristic approach** by achieving a better objective value (e.g., lower cost, shorter distance, higher efficiency).

    **Step 1: Analyze the Previous Heuristic Solution**
    - Review the last used heuristic and its results to determine if it **met the target goal**.
    - Identify why the previous solution was **not optimal**:
      - Did it fail to reach the given objective?
      - Were there inefficiencies in the heuristic approach?
      - Could computational performance be improved?
    - If the previous heuristic met the goal, determine if **further improvements are still possible**.
    - Identify **which parts of the previous approach were the main limitations** and require the most improvement.

    **Step 2: Generate an Improved Heuristic Solution**
    - Apply **a strictly improved heuristic approach** while ensuring that:
      - If the problem is a minimization problem (e.g., distance, cost), the new result **must be lower** than the previous one.
      - If the problem is a maximization problem (e.g., profit, efficiency), the new result **must be higher** than the previous one.
    - Explore different heuristic refinements:
      - **Enhance parameter tuning** (e.g., cooling schedules in simulated annealing, mutation rates in genetic algorithms).
      - **Use hybrid heuristics** (combine two or more techniques if beneficial).
      - **Introduce memory-based approaches** (e.g., Tabu Search to avoid cycling).
      - **Optimize computational efficiency** by improving runtime and memory usage.
    - Avoid unnecessary duplication of the previous code—focus on improvements.
    - Clearly document all improvements with inline comments explaining why changes were made.

    **Step 3: Validate and Compare the Improvement**
    - **Run a performance comparison** between the previous heuristic and the improved version.
    - If the new solution is **not strictly better**, retry optimization using a different heuristic refinement.
    - Ensure that the new heuristic is **computationally efficient** and does not introduce unnecessary complexity.
    - If the previous heuristic was **too simple**, explore **metaheuristics** (e.g., Simulated Annealing, Genetic Algorithms, Tabu Search).
    - If the solution is **not consistently better**, introduce an iterative improvement mechanism.

    **Step 4: Output the Best Heuristic Solution**
    Your response must include:

    - **python_code**: The improved heuristic-based Python code that strictly improves the previous result.
    - **requirements**: A list of all Python dependencies (e.g., numpy, scipy, networkx) needed to run the generated code.
    - **performance_comparison**: A summary of how the new heuristic solution compares to the previous one.
    - **objective_value**: The final computed value of the objective function in both the previous and new versions.
    - **test_cases**: A brief description of test cases used to verify that the new solution is indeed better.

    **Original user input:**
    {user_summary}

    **Original problem type:**
    {problem_type}

    **Optimization focus:**
    {optimization_focus}

    **Results from prior heuristic solutions:**
    {previous_results}
    
    **Last used heuristic-based code:**
    {previous_code}

    **Provided data (Python code, Excel files, VRP file, or may be empty):**
    {data}
    
    **Resource Requirements:**
    {resource_requirements}

    **Final Requirements**
    - Ensure the generated heuristic-based code **is fully functional, free from syntax errors, and improves on the previous version**.
    - The new objective value **must be strictly better** than the previous one.
    - If the new solution is **not better**, retry with a different heuristic refinement.
    - Ensure that the code outputs the **final computed objective value** clearly for easy comparison.
    - Automate testing: Generate test cases that verify the improvement.
    - Optimize runtime efficiency **without compromising solution quality**.
    """
)


FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the user's task:
    {user_input}

    And the following optimization codes:
    {summaries_of_optimizations}

    Your final report should include:

    1. **Performance Comparison**: Evaluate and compare the results of each optimization method, focusing on key metrics such as speed, memory usage, and task accuracy.
    2. **Best Optimization**: Identify the best optimization for the task, providing a clear explanation of why it is the most effective based on the comparison.
    3. **Visualization**: If applicable, include a visualization that clearly illustrates the performance differences between the optimizations.
    4. **Conclusion and Recommendations**: Summarize your findings and suggest potential improvements or note any limitations.

    The report should be concise, structured, and actionable, with a focus on addressing the user's specific task.
    """
)

CODE_FIXER_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an experienced software engineer specializing in debugging and fixing Python code. Your task is to analyze and resolve errors in the provided code, which failed to execute in a Docker container.

    **Step 1: Error Analysis**
    - Carefully examine the error message from the Docker logs and determine the **exact cause of failure**.
    - Categorize the error into one of the following types:
      - **SyntaxError / IndentationError** → Formatting or syntax issue.
      - **ImportError / ModuleNotFoundError** → Missing or incompatible package.
      - **TypeError / ValueError** → Incorrect data types or function usage.
      - **RuntimeError** → Execution-related failure (e.g., missing variable, infinite loop).
      - **EnvironmentError** → Issue related to Docker, file paths, or missing system dependencies.
    - Clearly explain why the error occurred before making any modifications.

    **Step 2: Code Fixing**
    - Make only **the minimal necessary modifications** to resolve the issue while maintaining the original logic.
    - If the error is related to **package compatibility**, suggest version adjustments in `requirements.txt`.
    - If an import is missing, determine whether it should be installed or if an alternative solution exists.
    - If the error is due to a **Docker-related constraint**, suggest a fix that works within the container environment.
    - Do **not modify external data files or access paths** unless strictly necessary.

    **Step 3: Validation**
    - Before returning the corrected code, ensure that it **executes successfully** in a simulated environment.
    - If the correction introduces new errors, attempt an alternative fix and update the response accordingly.

    **Docker Container Logs (Error Details)**:
    {docker_output}

    **Original Code**:
    {code}

    **Original Requirements**:
    {requirements}

    **Original Resources (Do not modify unless required)**:
    {resources}

    **Response Format**
    Return a JSON object with the following fields:
    - **error_analysis**: Explanation of why the error occurred.
    - **corrected_code**: The fixed version of the Python code.
    - **requirement_changes** (if applicable): Any adjustments made to package versions.
    - **execution_status**: Confirmation that the corrected code runs successfully.
    """
)
