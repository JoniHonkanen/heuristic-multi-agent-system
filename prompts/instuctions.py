CUTTING_STOCK = {
    "guidelines": """
General Guidelines for Cutting Stock Problems:

The cutting stock problem involves cutting large stock material (such as paper rolls or steel sheets) into smaller widths to fulfill customer orders, while minimizing material waste and the number of stock rolls used.

Key principles:

- Each cutting pattern defines how a single stock item (e.g., one 100-inch roll) is cut into a combination of required widths.
- The objective is to determine which patterns to use and how many times to apply each pattern so that demand for each width is fulfilled exactly.
- This is a variant of the one-dimensional bin packing problem with demand constraints.

Solution structure:

1. Pattern Generation:
   - Generate a set of feasible cutting patterns, where the sum of widths in each pattern does not exceed the total stock width (e.g., 100 inches).
   - Use constructive heuristics such as First-Fit Decreasing (FFD), Best-Fit Decreasing (BFD), or full enumeration with filtering.

2. Pattern Selection:
   - Determine how many times to use each pattern in order to meet all demands.
   - This can be done via linear or integer programming (e.g., using the 'pulp' or 'ortools' library), or with greedy or metaheuristic-based selection.
   - Metaheuristics such as Genetic Algorithms or Simulated Annealing may be used to improve efficiency in large instances.

The output must include:
- A list of all cutting patterns used
- The number of times each pattern is applied
- The total number of stock rolls used
- Optionally, the total material waste

Avoid the following:
- Assigning rolls per width individually. That does not reflect the nature of the problem.
- Overproducing or underproducing any width
- Using patterns that exceed the stock width limit

Recommended libraries:
- pandas, openpyxl: for Excel I/O
- pulp, ortools: for optimization
- collections, numpy: for data handling and pattern construction
""",
    "example": """
### Example – Heuristic Code Generation for Cutting Stock

Input:
<structured_state id="cutting-stock-1">
  <user_summary>Minimize the number of 100-inch rolls used to fulfill width demands</user_summary>
  <problem_type>Cutting stock</problem_type>
  <optimization_focus>Waste minimization</optimization_focus>
  <goal>Generate cutting patterns and select quantities</goal>
  <data>cutting_orders.xlsx</data>
  <resource_requirements>Widths: 45, 36, 31, 14; Demands: 97, 610, 395, 211</resource_requirements>
  <response_format>Excel</response_format>
</structured_state>

Output:
<assistant_response id="cutting-stock-1">
# Heuristic used: FFD pattern generation with greedy selection

from itertools import combinations_with_replacement

widths = [45, 36, 31, 14]
roll_width = 100
patterns = []

for r in range(1, 5):
    for combo in combinations_with_replacement(widths, r):
        if sum(combo) <= roll_width:
            patterns.append(combo)

</assistant_response>

Input:
<structured_state id="cutting-stock-2">
  <user_summary>Cut metal sheets efficiently using predefined widths</user_summary>
  <problem_type>Cutting stock</problem_type>
  <optimization_focus>Minimize number of rolls used</optimization_focus>
  <goal>Construct valid patterns for ILP solver</goal>
  <data>orders.csv</data>
  <resource_requirements>Widths and demands</resource_requirements>
  <response_format>CSV</response_format>
</structured_state>

Output:
<assistant_response id="cutting-stock-2">
# Heuristic used: Pattern generator for linear programming

def generate_patterns(widths, max_width):
    from itertools import combinations_with_replacement
    patterns = []
    for r in range(1, 5):
        for combo in combinations_with_replacement(widths, r):
            if sum(combo) <= max_width:
                patterns.append(sorted(combo))
    return list(set(map(tuple, patterns)))

</assistant_response>
""",
}

VRP = {
    "guidelines": """
=== General Guidelines for Vehicle Routing Problems (VRP) ===

The Vehicle Routing Problem (VRP) involves determining optimal delivery routes for a fleet of vehicles serving a set of customers, often from a central depot. The objective is typically to minimize total travel distance, cost, or time, while satisfying various constraints such as vehicle capacity.

You must use all input data exactly as provided. It is strictly forbidden to alter, restructure, simplify, or replace the input content in any way.

=== Input Structure ===

Typical VRP input includes:
- Customer locations and demands
- Distance or time matrix
- Vehicle capacity and fleet size
- Depot location

If input is given as a `.vrp` file or structured string, you must parse and use the data exactly as-is.
Replacing `.vrp` content with mockups, hardcoded strings, or altered versions is strictly forbidden.

=== Objective Requirements ===

Your implementation must:
- Serve each customer exactly once
- Assign customers to vehicle routes starting and ending at the depot
- Ensure no vehicle exceeds its capacity
- Generate a feasible route for each vehicle (if fleet size is fixed)
- Minimize total distance or cost

You must also clearly state the heuristic used and explain why it is appropriate for the problem.

=== Recommended Heuristics ===

- For small/medium VRPs:
  - Nearest Neighbor
  - Clarke-Wright Savings
  - Sweep Algorithm

- For larger/more complex cases:
  - Simulated Annealing
  - Tabu Search
  - Genetic Algorithms
  - Hyperheuristics
  - Google OR-Tools Routing Solver

=== Forbidden Practices ===

Never:
- Serve the same customer multiple times (unless explicitly allowed)
- Omit any customer
- Exceed vehicle capacity in any route
- Return raw distance matrices instead of structured routes
- Add units (e.g., km, kg) unless explicitly provided in the task

=== Recommended Libraries ===

- `ortools` (e.g., CVRP, VRPTW)
- `networkx`, `numpy` for graph and matrix handling
- `pandas`, `openpyxl` for I/O
- `matplotlib`, `folium` for optional visualization

=== Code Requirements ===

- All functions must be defined before they are used
- All required imports must be included
- Code must run in Python 3.9+ without modifications
- Only include packages actually used in `requirements.txt`
- Use flexible package versions (`>=x.y`), not exact pins (`==x.y`)

=== Output and Print Requirements ===

Your output must include:
- Total objective value (e.g., total route distance or cost)
- A human-readable route summary per vehicle
- Clear, valid print statements

Never do the following:
- Use `"text" + number"` — this is strictly forbidden and causes `TypeError`
- Insert raw newlines (`\n`) inside quoted strings — use escaped `\\n` only
- Use malformed `.format()` calls or unterminated strings

Correct:
    print(f"Truck 1 route: {{route1}}")
    print(f"Total distance: {{total_distance}}")

Incorrect:
    plan_output += ' {}\n\n'.format(x)  # Broken newline
    print("Distance: " + 123.4)         # TypeError

All output must be syntactically correct, safely formatted, and clearly convey the final result.
""",

    "example": """
### Example – Multi-vehicle routing using Clarke-Wright (VRPLIB input)

Input:
<structured_state id="vrp-1">
  <user_summary>Route optimization for 3 trucks delivering to 20 locations</user_summary>
  <problem_type>Vehicle routing</problem_type>
  <optimization_focus>Total distance minimization</optimization_focus>
  <goal>Generate optimized delivery routes</goal>
  <data>vrp_E-n20-k3.vrp</data>
  <resource_requirements>Truck cap: 100; Demands: per customer; Distance matrix included</resource_requirements>
  <response_format>Text summary + route details</response_format>
</structured_state>

Output:
<assistant_response id="vrp-1">
# Heuristic used: Clarke-Wright Savings Algorithm
# This heuristic computes "savings" for combining customer pairs into routes,
# starting from a separate route for each customer.

import pandas as pd
import networkx as nx

# Load and preprocess VRP data
# Parse .vrp file and extract depot, customers, capacities, and distance matrix

# Implement Clarke-Wright heuristic
# Build savings list and merge feasible routes

# Example output
print(f"Truck 1 route: [0, 3, 5, 8, 0]")
print(f"Truck 2 route: [0, 2, 7, 9, 0]")
print(f"Truck 3 route: [0, 4, 6, 1, 0]")
print(f"Total distance: 134.7")
</assistant_response>
"""
}

KNAPSACK = {
    "guidelines": """
General Guidelines for Knapsack Problems:

The Knapsack Problem is a classic combinatorial optimization task. The goal is to select a subset of items to maximize total value without exceeding a capacity constraint.

Problem structure:

1. Input includes:
   - A list of items with value and weight
   - A total capacity (e.g., weight limit)

2. Objective:
   - Maximize total value of selected items
   - Do not exceed the capacity constraint

Variants:
- 0/1 Knapsack: Each item can be selected at most once
- Fractional Knapsack: Items can be partially selected
- Multi-dimensional Knapsack: Multiple resource constraints
- Multiple knapsacks: Select items for several bags (e.g., bin packing)

Recommended heuristics:

- For 0/1 Knapsack:
  - Greedy heuristics based on value/weight ratio (approximate)
  - Dynamic programming (exact, but exponential in size)
  - Simulated Annealing, Genetic Algorithms for large instances

- For Fractional Knapsack:
  - Greedy method gives exact solution (value/weight sorted)

Avoid:
- Selecting items whose total weight exceeds the capacity
- Returning all items without applying any filtering or optimization
- Ignoring the value-to-weight trade-off

Recommended libraries:
- `pandas`, `numpy` for data handling
- `deap`, `random` for evolutionary/metaheuristics
- `pulp` or `ortools` for exact optimization if needed
""",
    "example": """
### Example – Heuristic Code Generation for Knapsack Problem

Input:
<structured_state id="knapsack-1">
  <user_summary>Select items to maximize value under a 50kg limit</user_summary>
  <problem_type>Knapsack</problem_type>
  <problem_class>knapsack</problem_class>
  <optimization_focus>Maximize value</optimization_focus>
  <goal>Heuristic-based item selection</goal>
  <data>items.csv</data>
  <resource_requirements>Each item has weight and value</resource_requirements>
  <response_format>List of selected items + total value</response_format>
</structured_state>

Output:
<assistant_response id="knapsack-1">
# Heuristic used: Greedy value/weight ratio

import pandas as pd

df = pd.read_csv("items.csv")
df["ratio"] = df["value"] / df["weight"]
df = df.sort_values(by="ratio", ascending=False)

capacity = 50
total_weight = 0
total_value = 0
selected = []

for _, row in df.iterrows():
    if total_weight + row["weight"] <= capacity:
        selected.append(row["item"])
        total_weight += row["weight"]
        total_value += row["value"]

print("Selected items:", selected)
print("Total value:", total_value)
</assistant_response>

Input:
<structured_state id="knapsack-2">
  <user_summary>Choose subset of products for delivery drone</user_summary>
  <problem_type>Knapsack</problem_type>
  <problem_class>knapsack</problem_class>
  <optimization_focus>Maximize delivered value</optimization_focus>
  <goal>Respect weight limit and select most valuable items</goal>
  <data>products.xlsx</data>
  <resource_requirements>Max weight 15kg</resource_requirements>
  <response_format>Excel file with selected items</response_format>
</structured_state>

Output:
<assistant_response id="knapsack-2">
# Heuristic used: Simulated Annealing for 0/1 Knapsack

# Initialize binary solution vector
# Perturb solution, evaluate, accept with probability
# Keep best-so-far solution based on value and weight feasibility
# Export selected items to Excel
</assistant_response>
""",
}

SHIFT_SCHEDULING = {
    "guidelines": """
General Guidelines for Shift Scheduling Problems:

The shift scheduling problem consists of assigning personnel to shifts over a planning horizon (e.g., a week or month), while satisfying a range of hard and soft constraints and optimizing objectives such as coverage, fairness, and preferences.

Typical input includes:
- A list of employees
- A calendar of shifts (e.g., morning, evening, night)
- Shift coverage requirements per day
- Rules or formulas for determining required staffing dynamically (e.g., based on number of clients, children, patients)
- Max/min working hours per employee
- Max consecutive working days
- Min rest between shifts (e.g., 11h EU directive)
- Employee qualifications or role constraints
- Preference scores, availability, or time-off requests

Objectives:
- Cover all required shifts
- Minimize constraint violations (e.g., rest time, max hours)
- Balance workload fairly
- Honor employee preferences as much as possible

Recommended approaches:
- Greedy construction with repair phase
- Local Search, Simulated Annealing, Tabu Search
- Integer Programming via `pulp` or `ortools` for small/medium instances
- Rule-based heuristics combined with preference scoring

Avoid:
- Assigning overlapping shifts to one employee
- Exceeding allowed working hours or violating minimum rest
- Leaving critical shifts uncovered
- Ignoring employee unavailability or time-off requests

Recommended libraries:
- `pandas`, `numpy` for data handling
- `ortools.sat.python.cp_model` for constraint-based solvers
- `random`, `datetime`, `calendar` for custom rule-based heuristics
""",
    "example": """
### Example – Heuristic Code Generation for Shift Scheduling

#### Simple example: Weekly rotation for nurses
Input:
<structured_state id="shift-1">
  <user_summary>Create a weekly schedule for 10 nurses</user_summary>
  <problem_type>Shift scheduling</problem_type>
  <problem_class>shift_scheduling</problem_class>
  <optimization_focus>Fair workload and rest compliance</optimization_focus>
  <goal>Assign each nurse to shifts for 7 days</goal>
  <data>nurses.csv</data>
  <resource_requirements>Morning/Evening/Night; max 40h/week; min 11h rest; no >5 consecutive days</resource_requirements>
  <response_format>Excel file with schedule</response_format>
</structured_state>

Output:
<assistant_response id="shift-1">
# Heuristic used: Rule-based greedy assignment with rest checks

import pandas as pd

nurses = pd.read_csv("nurses.csv")
schedule = []

# For each day and shift, assign first available nurse who:
# - Has not exceeded 40h
# - Has min 11h rest since last shift
# - Has worked ≤ 5 consecutive days

# Store schedule, export to Excel
</assistant_response>

#### Targeted example: Weekend assignments with availability constraints
Input:
<structured_state id="shift-2">
  <user_summary>Assign support staff to weekend shifts based on availability</user_summary>
  <problem_type>Shift scheduling</problem_type>
  <problem_class>shift_scheduling</problem_class>
  <optimization_focus>Minimize uncovered shifts and balance workload</optimization_focus>
  <goal>Generate fair weekend assignments</goal>
  <data>availability.xlsx</data>
  <resource_requirements>Support staff; 2 per shift; availability constraints</resource_requirements>
  <response_format>CSV with staff per shift</response_format>
</structured_state>

Output:
<assistant_response id="shift-2">
# Heuristic used: Availability-driven greedy selection + rotation fairness

# Read availability matrix
# For each weekend shift, assign available staff with lowest current load
# Track assigned hours per person
# Export CSV with per-shift assignments
</assistant_response>

#### Complex real-world case: Daycare center scheduling
Input:
<structured_state id="shift-3">
  <user_summary>Continue staff scheduling for a daycare center with 45 children and multiple child/staff ratios</user_summary>
  <problem_type>Shift scheduling</problem_type>
  <problem_class>shift_scheduling</problem_class>
  <optimization_focus>Coverage, group consistency, and workload balance</optimization_focus>
  <goal>Extend existing Excel-based schedule with valid weekly assignments</goal>
  <data>schedule_base.xlsx, child_data.xlsx, staff_profiles.xlsx</data>
  <resource_requirements>
  - 3 groups: Västäräkki, Peippola, Kiuru
  - 45 children total (15 under 3, 25 over 3, 5 mixed age)
  - Staff ratio: 1:4 (under 3), 1:7 (over 3), 1:1.75 (mixed), adjusted for special needs
  - Each staff has shift availability, group assignment, role, weekly target (38h 15min)
  - Shifts must respect holidays, avoid overlapping, and stay within hour limits
  </resource_requirements>
  <response_format>Excel file in same format as 'tyovuoro_esimerkki.xlsx' with assigned staff names per shift</response_format>
</structured_state>

Output:
<assistant_response id="shift-3">
# Heuristic used: Group-based demand estimation + constrained greedy assignment with load balancing

import pandas as pd

# Load data
schedule_template = pd.read_excel("schedule_base.xlsx")
child_data = pd.read_excel("child_data.xlsx")
staff_profiles = pd.read_excel("staff_profiles.xlsx")

# Step 1: Compute required staff per group per time slot based on attendance and adjusted ratios
# Step 2: For each shift slot:
#   - Select available staff from the correct group
#   - Ensure total hours and week limits are not exceeded
#   - Respect vacations and group assignment
# Step 3: Prioritize continuity and workload balance
# Step 4: Assign names to schedule cells, mark "loma" where needed

# Output schedule saved as Excel matching original template format ('tyovuoro_esimerkki.xlsx')

# Example (week 18, simplified preview):

| Week | Mon 6:30–14:30 | Mon 7:00–15:00 | Tue 6:30–14:30 | Tue 7:00–15:00 | ... |
|------|----------------|----------------|----------------|----------------|-----|
| 18   | Aalto, Tiina   | Korhonen, Liisa| Aalto          | Virtanen, Mika | ... |
| 19   | ...            | ...            | ...            | ...            |     |

# Final schedule exported to Excel as instructed.
</assistant_response>
""",
}

WAREHOUSE_OPTIMIZATION = {
"guidelines": """
General Guidelines for Warehouse Optimization Problems:

Warehouse optimization problems typically involve routing mobile agents (e.g., robots, workers) to pick up and deliver items between storage locations and designated areas, while minimizing total movement, time, or energy. Problems may also include optimizing the placement of goods to reduce future travel.

Typical input includes:
- A list of items with pickup and dropoff points
- A warehouse map represented as a grid or graph (e.g., aisles, shelves)
- Robot or agent profiles (e.g., capacity, speed, start location)
- Time windows or priority levels for some items
- Constraints like collision avoidance or one-way paths

Objectives:
- Minimize total distance or time used by robots
- Maximize throughput (items delivered per hour)
- Minimize idle time or energy consumption
- Balance task load among multiple robots

Constraints:
- No collisions or overlapping paths
- Maximum capacity per robot (e.g., 1–5 items per trip)
- Allowed directions in narrow aisles
- Limited shift duration or battery time
- Item priorities or dependencies (e.g., item A before B)

Recommended approaches:
- A* or Dijkstra for shortest pathfinding on warehouse graphs
- Multi-agent pathfinding (MAPF) for simultaneous routing
- Simulated Annealing or Genetic Algorithms for slot assignment or task sequencing
- Integer or Constraint Programming (e.g., with OR-Tools) for small-scale exact formulations
- Hybrid greedy + local search for fast approximations

Avoid:
- Treating robots as independent if collision constraints apply
- Ignoring start/end positions when optimizing sequence
- Overfitting to fixed item layouts
- Hard-coding shelf positions instead of using a flexible data structure

Recommended libraries:
- `networkx` or `scipy.sparse.csgraph` for graph-based routing
- `numpy`, `pandas` for data handling
- `ortools` for constraint and routing solvers
- `heapq`, `collections`, `matplotlib` for task queues and visualization
""",
"example": """
### Example – Heuristic Code Generation for Warehouse Routing

Input:
<structured_state id="warehouse-1">
  <user_summary>Minimize the movement of robots while delivering 10 items from entrance to shelf and from shelf to outbound dock</user_summary>
  <problem_type>Warehouse optimization</problem_type>
  <problem_class>warehouse_optimization</problem_class>
  <optimization_focus>Total distance traveled</optimization_focus>
  <goal>Move all items to their destination shelves with minimum motion</goal>
  <data>warehouse_map.json, items.csv</data>
  <resource_requirements>
  - Grid-based warehouse with shelves as nodes
  - 3 robots, each can carry 1 item at a time
  - Start location is (0, 0); outbound dock at (9, 9)
  - Items have storage targets like (2,3), (4,7), etc.
  </resource_requirements>
  <response_format>Python code that prints each robot's path and total cost</response_format>
</structured_state>

Output:
<assistant_response id="warehouse-1">
# Heuristic used: Task assignment + A* routing for each robot sequentially

import json, heapq
import networkx as nx
import pandas as pd

# Load warehouse layout
with open("warehouse_map.json") as f:
    warehouse = nx.node_link_graph(json.load(f))

items = pd.read_csv("items.csv")
robots = [{"id": i, "position": (0, 0), "path": [], "cost": 0} for i in range(3)]

# Assign each item to the robot with the lowest current load
for i, row in items.iterrows():
    dest = tuple(map(int, row["target"].split(",")))
    best_robot = min(robots, key=lambda r: r["cost"])
    path = nx.shortest_path(warehouse, source=best_robot["position"], target=dest, weight="weight")
    best_robot["path"] += path + [dest]
    best_robot["cost"] += nx.path_weight(warehouse, path, weight="weight")
    best_robot["position"] = dest

# Move to dock after deliveries
for robot in robots:
    path = nx.shortest_path(warehouse, source=robot["position"], target=(9, 9), weight="weight")
    robot["path"] += path
    robot["cost"] += nx.path_weight(warehouse, path, weight="weight")

# Print result
for r in robots:
    print(f"Robot {r['id']}: Path = {r['path']}, Cost = {r['cost']}")
</assistant_response>
"""
}

OTHER = {
    "guidelines": """
No specific problem class was identified. Apply general best practices for heuristic-based optimization.

Recommended heuristics by problem family:

- VRP / TSP: Nearest Neighbor, Clarke-Wright, Simulated Annealing, Tabu Search, Genetic Algorithms, Ant Colony Optimization
- Cutting stock: First-Fit Decreasing (FFD), Best-Fit Decreasing (BFD), Simulated Annealing, Genetic Algorithms
- Scheduling: Iterated Local Search (ILS), Variable Neighborhood Search (VNS), Tabu Search, Hyperheuristics
- Multi-objective: NSGA-II (Non-dominated Sorting Genetic Algorithm II)

General advice:
- Use greedy construction methods when applicable
- Combine with local improvement or metaheuristics
- Use problem-specific constraints to prune the solution space

Avoid:
- Applying complex metaheuristics without justification
- Ignoring constraints or objective definition
- Generating trivial solutions (e.g., all items selected, no optimization)
""",
    "example": """
### Example – Heuristic Code Generation (Generic)

Input:
<structured_state id="example-1">
  <user_summary>Allocate resources to minimize total cost across tasks</user_summary>
  <problem_type>Resource allocation</problem_type>
  <problem_class>other</problem_class>
  <optimization_focus>Cost minimization</optimization_focus>
  <goal>Heuristic-based assignment</goal>
  <data>task_input.csv</data>
  <resource_requirements>Task costs and limits</resource_requirements>
  <response_format>CSV</response_format>
</structured_state>

Output:
<assistant_response id="example-1">
# Heuristic used: Greedy allocation based on cost-benefit ratio

import pandas as pd

df = pd.read_csv("task_input.csv")
df["ratio"] = df["benefit"] / df["cost"]
df = df.sort_values(by="ratio", ascending=False)

# Allocate resources to tasks with highest benefit/cost first
selected = []
total_cost = 0
budget = 100

for _, row in df.iterrows():
    if total_cost + row["cost"] <= budget:
        selected.append(row["task_id"])
        total_cost += row["cost"]

print("Selected tasks:", selected)
print("Total cost:", total_cost)
</assistant_response>
""",
}
