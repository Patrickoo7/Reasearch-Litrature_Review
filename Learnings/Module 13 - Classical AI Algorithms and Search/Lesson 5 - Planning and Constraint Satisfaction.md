# Lesson 5: Planning & Constraint Satisfaction 🤖

**Module 13: Classical AI Algorithms and Search | Lesson 5 of 5**

Master automated planning and constraint solving - from Sudoku to robot mission planning!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand STRIPS planning representation
2. ✅ Implement forward and backward planning algorithms
3. ✅ Master Constraint Satisfaction Problems (CSP)
4. ✅ Build backtracking search with constraint propagation
5. ✅ Apply CSP to real problems (Sudoku, scheduling, map coloring)
6. ✅ Understand heuristics for CSP solving

---

## 1. Introduction to Planning

### What is Automated Planning?

**Planning** finds a sequence of actions to achieve a goal from an initial state.

**Key Differences from Search:**
- **Search**: Explicitly enumerate states
- **Planning**: Use structured representations (actions, preconditions, effects)

```python
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from copy import deepcopy
from collections import deque

class State:
    """
    State representation using propositions.

    Example: State({'at(robot, A)', 'box(box1, B)', 'empty(C)'})
    """
    def __init__(self, propositions: Set[str]):
        self.propositions = frozenset(propositions)

    def holds(self, proposition: str) -> bool:
        """Check if proposition is true in this state."""
        return proposition in self.propositions

    def satisfies(self, propositions: Set[str]) -> bool:
        """Check if all propositions are true."""
        return propositions.issubset(self.propositions)

    def apply_effects(self, add_effects: Set[str], delete_effects: Set[str]):
        """Apply action effects to create new state."""
        new_props = (self.propositions - delete_effects) | add_effects
        return State(new_props)

    def __eq__(self, other):
        return self.propositions == other.propositions

    def __hash__(self):
        return hash(self.propositions)

    def __repr__(self):
        return f"State({sorted(self.propositions)})"


class Action:
    """
    Action in STRIPS representation.

    Components:
    - name: Action identifier
    - preconditions: Propositions that must be true
    - add_effects: Propositions that become true
    - delete_effects: Propositions that become false
    """
    def __init__(self, name: str,
                 preconditions: Set[str],
                 add_effects: Set[str],
                 delete_effects: Set[str]):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.delete_effects = frozenset(delete_effects)

    def is_applicable(self, state: State) -> bool:
        """Check if action can be executed in state."""
        return state.satisfies(self.preconditions)

    def apply(self, state: State) -> State:
        """Execute action and return new state."""
        if not self.is_applicable(state):
            raise ValueError(f"Action {self.name} not applicable in {state}")

        return state.apply_effects(self.add_effects, self.delete_effects)

    def __repr__(self):
        return self.name


# Example: Blocks World
print("="*60)
print("STRIPS Planning - Blocks World")
print("="*60)

# Initial state: A on table, B on A, C on table
initial_state = State({
    'on(A, Table)',
    'on(B, A)',
    'on(C, Table)',
    'clear(B)',
    'clear(C)',
    'handempty'
})

# Goal state: B on table, A on B, C on A
goal_state = State({
    'on(B, Table)',
    'on(A, B)',
    'on(C, A)',
    'clear(C)',
    'handempty'
})

# Define actions
def create_blocks_world_actions():
    """Create actions for blocks world."""
    actions = []

    blocks = ['A', 'B', 'C']

    for block in blocks:
        # Pickup from table
        pickup = Action(
            name=f'pickup({block})',
            preconditions={f'on({block}, Table)', f'clear({block})', 'handempty'},
            add_effects={f'holding({block})'},
            delete_effects={f'on({block}, Table)', f'clear({block})', 'handempty'}
        )
        actions.append(pickup)

        # Putdown on table
        putdown = Action(
            name=f'putdown({block})',
            preconditions={f'holding({block})'},
            add_effects={f'on({block}, Table)', f'clear({block})', 'handempty'},
            delete_effects={f'holding({block})'}
        )
        actions.append(putdown)

        # Stack block1 on block2
        for other_block in blocks:
            if block != other_block:
                stack = Action(
                    name=f'stack({block}, {other_block})',
                    preconditions={f'holding({block})', f'clear({other_block})'},
                    add_effects={f'on({block}, {other_block})', f'clear({block})', 'handempty'},
                    delete_effects={f'holding({block})', f'clear({other_block})'}
                )
                actions.append(stack)

                # Unstack block1 from block2
                unstack = Action(
                    name=f'unstack({block}, {other_block})',
                    preconditions={f'on({block}, {other_block})', f'clear({block})', 'handempty'},
                    add_effects={f'holding({block})', f'clear({other_block})'},
                    delete_effects={f'on({block}, {other_block})', f'clear({block})', 'handempty'}
                )
                actions.append(unstack)

    return actions


actions = create_blocks_world_actions()
print(f"\nCreated {len(actions)} actions")
print("Sample actions:", [a.name for a in actions[:3]])

print(f"\nInitial state: {initial_state}")
print(f"Goal state: {goal_state}")
```

---

## 2. Forward Planning

### Forward Search in State Space

```python
def forward_planning(initial_state: State,
                    goal_propositions: Set[str],
                    actions: List[Action],
                    max_depth: int = 20) -> Optional[List[Action]]:
    """
    Forward planning using breadth-first search.

    Search from initial state towards goal.

    Returns:
        plan: Sequence of actions to reach goal
    """
    # BFS in state space
    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        state, plan = queue.popleft()

        # Check if goal reached
        if state.satisfies(goal_propositions):
            return plan

        # Don't search too deep
        if len(plan) >= max_depth:
            continue

        # Try all applicable actions
        for action in actions:
            if action.is_applicable(state):
                new_state = action.apply(state)

                if new_state not in visited:
                    visited.add(new_state)
                    new_plan = plan + [action]
                    queue.append((new_state, new_plan))

    return None  # No plan found


# Run forward planning
print("\n" + "="*60)
print("Forward Planning")
print("="*60)

goal_props = {
    'on(B, Table)',
    'on(A, B)',
    'on(C, A)',
    'clear(C)',
    'handempty'
}

plan = forward_planning(initial_state, goal_props, actions, max_depth=15)

if plan:
    print(f"\nFound plan with {len(plan)} steps:")
    for i, action in enumerate(plan, 1):
        print(f"  {i}. {action.name}")

    # Execute plan to verify
    state = initial_state
    print("\nExecuting plan:")
    print(f"  Initial: {state}")

    for action in plan:
        state = action.apply(state)
        print(f"  After {action.name}:")
        print(f"    {state}")

    print(f"\n✅ Goal reached: {state.satisfies(goal_props)}")
else:
    print("No plan found!")
```

---

## 3. Backward Planning

### Backward Search (Goal Regression)

```python
def backward_planning(initial_state: State,
                     goal_propositions: Set[str],
                     actions: List[Action],
                     max_depth: int = 20) -> Optional[List[Action]]:
    """
    Backward planning using goal regression.

    Search from goal towards initial state.

    Returns:
        plan: Sequence of actions to reach goal
    """
    def regress_goal(goal: Set[str], action: Action) -> Optional[Set[str]]:
        """
        Compute preconditions needed before action.

        New goal = (goal - add_effects) ∪ preconditions
        """
        # Check if action is relevant (adds something we need)
        if not (goal & action.add_effects):
            return None

        # Check if action doesn't delete something we need
        if goal & action.delete_effects:
            return None

        # Regress goal through action
        new_goal = (goal - action.add_effects) | action.preconditions
        return new_goal

    # BFS in goal space
    queue = deque([(frozenset(goal_propositions), [])])
    visited = {frozenset(goal_propositions)}

    while queue:
        current_goal, plan = queue.popleft()

        # Check if initial state satisfies current goal
        if initial_state.satisfies(current_goal):
            # Reverse plan (we built it backwards)
            return list(reversed(plan))

        if len(plan) >= max_depth:
            continue

        # Try regressing through each action
        for action in actions:
            new_goal = regress_goal(current_goal, action)

            if new_goal is not None and new_goal not in visited:
                visited.add(new_goal)
                new_plan = plan + [action]
                queue.append((new_goal, new_plan))

    return None


# Run backward planning
print("\n" + "="*60)
print("Backward Planning")
print("="*60)

plan_backward = backward_planning(initial_state, goal_props, actions, max_depth=15)

if plan_backward:
    print(f"\nFound plan with {len(plan_backward)} steps:")
    for i, action in enumerate(plan_backward, 1):
        print(f"  {i}. {action.name}")
else:
    print("No plan found!")
```

---

## 4. Constraint Satisfaction Problems (CSP)

### CSP Fundamentals

A CSP consists of:
- **Variables**: X₁, X₂, ..., Xₙ
- **Domains**: D₁, D₂, ..., Dₙ (possible values)
- **Constraints**: Relations between variables

```python
class CSP:
    """
    Constraint Satisfaction Problem.

    Variables have domains, and constraints restrict values.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List],
                 constraints: List):
        """
        Args:
            variables: List of variable names
            domains: Dict {variable: [possible_values]}
            constraints: List of constraint functions
        """
        self.variables = variables
        self.domains = {var: list(domain) for var, domain in domains.items()}
        self.constraints = constraints

        # Current assignment
        self.assignment = {}

    def is_consistent(self, var, value, assignment):
        """
        Check if assigning value to var is consistent with assignment.

        Tests all constraints involving var.
        """
        # Temporary assignment
        temp_assignment = assignment.copy()
        temp_assignment[var] = value

        # Check all constraints
        for constraint in self.constraints:
            if not constraint(temp_assignment):
                return False

        return True

    def is_complete(self, assignment):
        """Check if all variables are assigned."""
        return len(assignment) == len(self.variables)

    def select_unassigned_variable(self, assignment):
        """Select next variable to assign (simple: first unassigned)."""
        for var in self.variables:
            if var not in assignment:
                return var
        return None

    def order_domain_values(self, var, assignment):
        """Order domain values for variable (simple: original order)."""
        return self.domains[var]


# Example: Map Coloring
print("\n" + "="*60)
print("CSP: Map Coloring Problem")
print("="*60)

# Variables: Australian states
variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

# Domains: Colors
colors = ['red', 'green', 'blue']
domains = {var: colors for var in variables}

# Constraints: Adjacent regions must have different colors
def make_not_equal_constraint(var1, var2):
    """Create constraint that var1 != var2."""
    def constraint(assignment):
        if var1 in assignment and var2 in assignment:
            return assignment[var1] != assignment[var2]
        return True
    return constraint


# Define adjacencies
adjacencies = [
    ('WA', 'NT'), ('WA', 'SA'),
    ('NT', 'SA'), ('NT', 'Q'),
    ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
    ('Q', 'NSW'),
    ('NSW', 'V')
]

constraints = [make_not_equal_constraint(v1, v2) for v1, v2 in adjacencies]

map_csp = CSP(variables, domains, constraints)

print(f"Variables: {variables}")
print(f"Domain size: {len(colors)} colors")
print(f"Constraints: {len(constraints)}")
```

---

## 5. Backtracking Search

### Basic Backtracking

```python
def backtracking_search(csp: CSP) -> Optional[Dict]:
    """
    Backtracking search for CSP.

    Recursively assign values to variables.
    Backtrack when constraints violated.

    Returns:
        assignment: Dict {variable: value}
    """
    def backtrack(assignment):
        # Check if complete
        if csp.is_complete(assignment):
            return assignment

        # Select unassigned variable
        var = csp.select_unassigned_variable(assignment)

        # Try each value in domain
        for value in csp.order_domain_values(var, assignment):
            # Check consistency
            if csp.is_consistent(var, value, assignment):
                # Assign value
                assignment[var] = value

                # Recurse
                result = backtrack(assignment)

                if result is not None:
                    return result

                # Backtrack
                del assignment[var]

        return None

    return backtrack({})


# Solve map coloring
print("\n" + "="*60)
print("Backtracking Search")
print("="*60)

solution = backtracking_search(map_csp)

if solution:
    print("\n✅ Solution found:")
    for var in sorted(solution.keys()):
        print(f"  {var}: {solution[var]}")
else:
    print("❌ No solution found")
```

---

## 6. Constraint Propagation

### Arc Consistency (AC-3)

```python
class CSPWithArcConsistency(CSP):
    """
    CSP with arc consistency preprocessing.

    AC-3 algorithm removes values from domains that can never
    be part of a solution.
    """
    def __init__(self, variables, domains, constraints):
        super().__init__(variables, domains, constraints)
        self.constraint_graph = self._build_constraint_graph()

    def _build_constraint_graph(self):
        """Build graph of which variables constrain each other."""
        graph = {var: set() for var in self.variables}

        for constraint in self.constraints:
            # Extract variables from constraint (simplified)
            # In practice, would need to analyze constraint function
            pass

        return graph

    def revise(self, xi, xj):
        """
        Make xi arc-consistent with xj.

        Remove values from D_i that have no support in D_j.
        """
        revised = False

        # For each value in xi's domain
        for value_i in self.domains[xi][:]:  # Copy to allow modification
            # Check if there exists a value in xj's domain that satisfies constraints
            has_support = False

            for value_j in self.domains[xj]:
                # Check if (xi=value_i, xj=value_j) satisfies all constraints
                temp_assignment = {xi: value_i, xj: value_j}

                # Check all constraints involving both variables
                consistent = True
                for constraint in self.constraints:
                    if not constraint(temp_assignment):
                        consistent = False
                        break

                if consistent:
                    has_support = True
                    break

            # If no support, remove value
            if not has_support:
                self.domains[xi].remove(value_i)
                revised = True

        return revised

    def ac3(self):
        """
        AC-3 algorithm for arc consistency.

        Returns:
            True if consistent, False if domain becomes empty
        """
        # Initialize queue with all arcs
        queue = deque()
        for constraint in self.constraints:
            # Add both directions for each constraint
            # (Simplified: assumes binary constraints)
            pass

        # Simplified: Just iterate over all variable pairs
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 != var2:
                    queue.append((var1, var2))

        while queue:
            xi, xj = queue.popleft()

            if self.revise(xi, xj):
                # Domain of xi changed
                if len(self.domains[xi]) == 0:
                    return False  # No solution

                # Add all arcs (xk, xi) for neighbors xk of xi
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        return True


# Test AC-3
print("\n" + "="*60)
print("Arc Consistency (AC-3)")
print("="*60)

csp_ac = CSPWithArcConsistency(variables, domains, constraints)

print("Domain sizes before AC-3:")
for var in variables:
    print(f"  {var}: {len(csp_ac.domains[var])} values")

# Run AC-3
consistent = csp_ac.ac3()

print(f"\nAC-3 result: {'Consistent' if consistent else 'Inconsistent'}")

if consistent:
    print("\nDomain sizes after AC-3:")
    for var in variables:
        print(f"  {var}: {len(csp_ac.domains[var])} values")
```

---

## 7. CSP Heuristics

### Variable and Value Ordering

```python
class ImprovedCSP(CSP):
    """CSP with intelligent variable and value ordering."""

    def select_unassigned_variable(self, assignment):
        """
        Minimum Remaining Values (MRV) heuristic.

        Choose variable with fewest legal values remaining.
        Also called "most constrained variable" heuristic.
        """
        unassigned = [v for v in self.variables if v not in assignment]

        def count_legal_values(var):
            count = 0
            for value in self.domains[var]:
                if self.is_consistent(var, value, assignment):
                    count += 1
            return count

        # Choose variable with minimum legal values
        return min(unassigned, key=count_legal_values)

    def order_domain_values(self, var, assignment):
        """
        Least Constraining Value (LCV) heuristic.

        Try values that rule out fewest choices for neighbors.
        """
        def count_conflicts(value):
            # Count how many values this removes from neighbors
            conflicts = 0
            temp_assignment = assignment.copy()
            temp_assignment[var] = value

            for other_var in self.variables:
                if other_var != var and other_var not in assignment:
                    for other_value in self.domains[other_var]:
                        if not self.is_consistent(other_var, other_value, temp_assignment):
                            conflicts += 1

            return conflicts

        # Sort by increasing conflicts
        return sorted(self.domains[var], key=count_conflicts)


# Test improved CSP
print("\n" + "="*60)
print("Improved CSP with MRV and LCV")
print("="*60)

improved_csp = ImprovedCSP(variables, domains, constraints)
solution_improved = backtracking_search(improved_csp)

if solution_improved:
    print("\n✅ Solution found with heuristics:")
    for var in sorted(solution_improved.keys()):
        print(f"  {var}: {solution_improved[var]}")
```

---

## 8. Real-World Applications

### Application 1: Sudoku Solver

```python
class SudokuCSP(CSP):
    """
    Sudoku as a CSP.

    Variables: 81 cells
    Domain: {1, 2, ..., 9}
    Constraints: All-different in rows, columns, boxes
    """
    def __init__(self, puzzle):
        """
        Args:
            puzzle: 9x9 array (0 = empty)
        """
        self.puzzle = np.array(puzzle)

        # Variables: (row, col) for each cell
        variables = [(i, j) for i in range(9) for j in range(9)]

        # Domains: 1-9 for empty cells, fixed value for given cells
        domains = {}
        for i in range(9):
            for j in range(9):
                if puzzle[i][j] == 0:
                    domains[(i, j)] = list(range(1, 10))
                else:
                    domains[(i, j)] = [puzzle[i][j]]

        # Constraints: All-different in rows, columns, 3x3 boxes
        constraints = []

        # Row constraints
        for i in range(9):
            for j1 in range(9):
                for j2 in range(j1 + 1, 9):
                    constraints.append(
                        make_not_equal_constraint((i, j1), (i, j2))
                    )

        # Column constraints
        for j in range(9):
            for i1 in range(9):
                for i2 in range(i1 + 1, 9):
                    constraints.append(
                        make_not_equal_constraint((i1, j), (i2, j))
                    )

        # Box constraints
        for box_i in range(3):
            for box_j in range(3):
                cells = [
                    (box_i * 3 + i, box_j * 3 + j)
                    for i in range(3) for j in range(3)
                ]
                for idx1 in range(len(cells)):
                    for idx2 in range(idx1 + 1, len(cells)):
                        constraints.append(
                            make_not_equal_constraint(cells[idx1], cells[idx2])
                        )

        super().__init__(variables, domains, constraints)

    def assignment_to_grid(self, assignment):
        """Convert assignment to 9x9 grid."""
        grid = np.zeros((9, 9), dtype=int)
        for (i, j), value in assignment.items():
            grid[i][j] = value
        return grid


# Solve Sudoku
print("\n" + "="*60)
print("Sudoku Solver")
print("="*60)

# Easy Sudoku puzzle
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

print("Puzzle:")
for row in puzzle:
    print("  ", " ".join(str(x) if x != 0 else "." for x in row))

sudoku = SudokuCSP(puzzle)
solution = backtracking_search(sudoku)

if solution:
    print("\n✅ Solution:")
    grid = sudoku.assignment_to_grid(solution)
    for row in grid:
        print("  ", " ".join(str(x) for x in row))
else:
    print("❌ No solution found")
```

### Application 2: N-Queens Problem

```python
class NQueensCSP(CSP):
    """
    N-Queens problem as CSP.

    Place N queens on NxN board so no two queens attack each other.
    """
    def __init__(self, n):
        """
        Args:
            n: Board size and number of queens
        """
        self.n = n

        # Variables: Queen in each row
        # Variable i represents row i
        variables = list(range(n))

        # Domain: Column for each queen
        domains = {i: list(range(n)) for i in range(n)}

        # Constraints: No two queens in same column or diagonal
        constraints = []

        for i in range(n):
            for j in range(i + 1, n):
                # Different columns
                def make_column_constraint(row1, row2):
                    def constraint(assignment):
                        if row1 in assignment and row2 in assignment:
                            return assignment[row1] != assignment[row2]
                        return True
                    return constraint

                constraints.append(make_column_constraint(i, j))

                # Different diagonals
                def make_diagonal_constraint(row1, row2):
                    def constraint(assignment):
                        if row1 in assignment and row2 in assignment:
                            col1 = assignment[row1]
                            col2 = assignment[row2]
                            # Not on same diagonal
                            return abs(row1 - row2) != abs(col1 - col2)
                        return True
                    return constraint

                constraints.append(make_diagonal_constraint(i, j))

        super().__init__(variables, domains, constraints)

    def visualize_solution(self, assignment):
        """Print board with queens."""
        board = [['.' for _ in range(self.n)] for _ in range(self.n)]

        for row, col in assignment.items():
            board[row][col] = 'Q'

        for row in board:
            print("  ", " ".join(row))


# Solve N-Queens
print("\n" + "="*60)
print("N-Queens Problem")
print("="*60)

for n in [4, 8]:
    print(f"\n{n}-Queens:")
    nqueens = NQueensCSP(n)
    solution = backtracking_search(nqueens)

    if solution:
        print("✅ Solution found:")
        nqueens.visualize_solution(solution)
    else:
        print("❌ No solution")
```

### Application 3: Course Scheduling

```python
class CourseSchedulingCSP(CSP):
    """
    University course scheduling.

    Assign courses to time slots and rooms.
    """
    def __init__(self, courses, time_slots, rooms, constraints_data):
        """
        Args:
            courses: List of course IDs
            time_slots: List of time slot IDs
            rooms: List of room IDs
            constraints_data: Dict with constraint info
        """
        # Variables: Each course needs (time, room)
        variables = courses

        # Domain: (time_slot, room) pairs
        domains = {
            course: [(t, r) for t in time_slots for r in rooms]
            for course in courses
        }

        # Constraints
        constraints = []

        # No two courses in same room at same time
        for i, course1 in enumerate(courses):
            for course2 in courses[i+1:]:
                def make_room_time_constraint(c1, c2):
                    def constraint(assignment):
                        if c1 in assignment and c2 in assignment:
                            t1, r1 = assignment[c1]
                            t2, r2 = assignment[c2]
                            # Different room OR different time
                            return r1 != r2 or t1 != t2
                        return True
                    return constraint

                constraints.append(make_room_time_constraint(course1, course2))

        # Instructor constraints (same instructor can't teach two courses at once)
        instructor_courses = constraints_data.get('instructor_courses', {})
        for instructor, course_list in instructor_courses.items():
            for i, c1 in enumerate(course_list):
                for c2 in course_list[i+1:]:
                    def make_instructor_constraint(course1, course2):
                        def constraint(assignment):
                            if course1 in assignment and course2 in assignment:
                                t1, r1 = assignment[course1]
                                t2, r2 = assignment[course2]
                                return t1 != t2  # Different times
                            return True
                        return constraint

                    constraints.append(make_instructor_constraint(c1, c2))

        super().__init__(variables, domains, constraints)


# Course scheduling example
print("\n" + "="*60)
print("Course Scheduling")
print("="*60)

courses = ['CS101', 'CS102', 'MATH101', 'PHYS101']
time_slots = ['Mon-9AM', 'Mon-11AM', 'Tue-9AM', 'Tue-11AM']
rooms = ['Room-A', 'Room-B']

constraints_data = {
    'instructor_courses': {
        'Prof-Smith': ['CS101', 'CS102'],
        'Prof-Jones': ['MATH101'],
        'Prof-Brown': ['PHYS101']
    }
}

scheduling = CourseSchedulingCSP(courses, time_slots, rooms, constraints_data)
schedule = backtracking_search(scheduling)

if schedule:
    print("\n✅ Schedule found:")
    for course in sorted(schedule.keys()):
        time, room = schedule[course]
        print(f"  {course:10s}: {time:12s} in {room}")
else:
    print("❌ No valid schedule")
```

---

## 9. Advanced Planning Techniques

### Hierarchical Task Network (HTN) Planning

```python
class HTNPlanner:
    """
    Hierarchical Task Network planning.

    Decompose high-level tasks into subtasks.
    """
    def __init__(self):
        self.methods = {}  # {task: [decompositions]}
        self.primitive_actions = {}  # {action: (preconditions, effects)}

    def add_method(self, task, subtasks, preconditions=None):
        """
        Add decomposition method for task.

        Args:
            task: High-level task name
            subtasks: List of subtasks
            preconditions: Conditions for this decomposition
        """
        if task not in self.methods:
            self.methods[task] = []

        self.methods[task].append({
            'subtasks': subtasks,
            'preconditions': preconditions or set()
        })

    def add_primitive_action(self, action, preconditions, effects):
        """Add primitive action (executable)."""
        self.primitive_actions[action] = {
            'preconditions': preconditions,
            'effects': effects
        }

    def plan(self, tasks, state):
        """
        Plan for list of tasks.

        Returns:
            plan: List of primitive actions
        """
        if not tasks:
            return []

        task = tasks[0]
        remaining_tasks = tasks[1:]

        # If primitive action
        if task in self.primitive_actions:
            action_info = self.primitive_actions[task]

            # Check preconditions
            if state.satisfies(action_info['preconditions']):
                # Apply effects
                new_state = state.apply_effects(
                    action_info['effects'].get('add', set()),
                    action_info['effects'].get('delete', set())
                )

                # Plan for remaining tasks
                rest_plan = self.plan(remaining_tasks, new_state)

                if rest_plan is not None:
                    return [task] + rest_plan

            return None

        # If compound task, try decompositions
        if task in self.methods:
            for method in self.methods[task]:
                # Check method preconditions
                if state.satisfies(method['preconditions']):
                    # Decompose
                    new_tasks = method['subtasks'] + remaining_tasks

                    # Recursively plan
                    plan = self.plan(new_tasks, state)

                    if plan is not None:
                        return plan

        return None


# HTN planning example
print("\n" + "="*60)
print("Hierarchical Task Network Planning")
print("="*60)

htn = HTNPlanner()

# Define primitive actions
htn.add_primitive_action(
    'get_taxi',
    preconditions={'at(home)'},
    effects={'add': {'have(taxi)'}, 'delete': set()}
)

htn.add_primitive_action(
    'ride_taxi',
    preconditions={'have(taxi)'},
    effects={'add': {'at(airport)'}, 'delete': {'at(home)', 'have(taxi)'}}
)

htn.add_primitive_action(
    'buy_ticket',
    preconditions={'at(airport)'},
    effects={'add': {'have(ticket)'}, 'delete': set()}
)

# Define compound task
htn.add_method(
    'travel_to_airport',
    subtasks=['get_taxi', 'ride_taxi'],
    preconditions={'at(home)'}
)

htn.add_method(
    'prepare_for_flight',
    subtasks=['travel_to_airport', 'buy_ticket'],
    preconditions={'at(home)'}
)

# Plan
initial = State({'at(home)', 'have(money)'})
tasks = ['prepare_for_flight']

plan = htn.plan(tasks, initial)

if plan:
    print("\n✅ HTN Plan:")
    for i, action in enumerate(plan, 1):
        print(f"  {i}. {action}")
else:
    print("❌ No plan found")
```

---

## 10. Practice Exercises

### Exercise 1: Implement GRAPHPLAN

```python
"""
Implement GRAPHPLAN algorithm.

GRAPHPLAN builds a planning graph:
- Alternating layers of propositions and actions
- Mutual exclusion (mutex) relations
- Extract plan by backward search through graph

More efficient than forward/backward search for many problems.
"""

# Your implementation here
```

### Exercise 2: Implement Min-Conflicts

```python
"""
Implement Min-Conflicts algorithm for CSP.

Local search approach:
1. Start with random complete assignment (may violate constraints)
2. Repeat: Pick conflicted variable, reassign to minimize conflicts
3. Stop when no conflicts or max iterations

Very effective for large CSPs (e.g., million queens problem).
"""

# Your implementation here
```

### Exercise 3: Build Job Shop Scheduler

```python
"""
Build job shop scheduling system using CSP.

Problem:
- N jobs, each with sequence of tasks
- Each task requires specific machine for certain duration
- Minimize makespan (total completion time)

Constraints:
- Tasks in same job must be sequential
- Each machine handles one task at a time
- Precedence constraints

Apply to: Manufacturing, cloud computing, project management
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **STRIPS Planning** 🎯
   - State: Set of propositions
   - Actions: Preconditions + effects
   - Forward search: Start → goal
   - Backward search: Goal → start

2. **Planning vs Search** 🔍
   - Planning: Structured representation
   - More efficient than state enumeration
   - Can exploit problem structure

3. **Constraint Satisfaction Problems** 🧩
   - Variables, domains, constraints
   - General framework for many problems
   - Backtracking search as baseline

4. **Constraint Propagation** ⚡
   - Arc consistency (AC-3)
   - Reduces search space
   - Detects inconsistencies early

5. **CSP Heuristics** 🎨
   - **MRV**: Most constrained variable first
   - **LCV**: Least constraining value first
   - Dramatically improve performance

### Algorithm Comparison

| Algorithm | Problem Type | Completeness | Optimality |
|-----------|-------------|--------------|------------|
| Forward Planning | Goal achievement | Yes | Yes (with optimal search) |
| Backward Planning | Goal achievement | Yes | Yes (with optimal search) |
| Backtracking | CSP | Yes | Yes (finds satisfying assignment) |
| AC-3 | CSP preprocessing | - | - (not search algorithm) |
| HTN Planning | Hierarchical tasks | Yes | Depends on methods |

### Real-World Applications

✅ **Robot Planning**: Motion planning, task planning
✅ **Sudoku/Puzzles**: CSP formulation
✅ **Scheduling**: Course scheduling, job shop scheduling
✅ **Configuration**: Product configuration (e.g., car options)
✅ **Resource Allocation**: Assigning tasks to workers
✅ **Circuit Design**: VLSI layout

### What's Next?

You've completed Module 13! You've mastered:
- Search algorithms (A*, beam search)
- Game playing (minimax, MCTS, AlphaGo)
- Optimization (GA, SA, PSO, ACO)
- Graph algorithms (PageRank, shortest paths)
- Planning and CSP

**Next Steps:**
- Apply these to your own projects
- Combine with deep learning (AlphaGo = MCTS + RL)
- Explore advanced planning (temporal planning, probabilistic planning)
- Study modern CSP solvers (SAT, SMT solvers)

**Classical AI algorithms are timeless - they power everything from GPS to Google!** 🚀

---

## Additional Resources

### Papers
- Fikes & Nilsson (1971): "STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving"
- Blum & Furst (1997): "Fast Planning Through Planning Graph Analysis" (GRAPHPLAN)
- Mackworth (1977): "Consistency in Networks of Relations" (Arc Consistency)

### Books
- **Artificial Intelligence: A Modern Approach** (Russell & Norvig) - Chapters 10-11
- **Automated Planning: Theory and Practice** (Ghallab, Nau, Traverso)
- **Constraint Processing** (Dechter)

### Libraries
- **python-constraint**: CSP library for Python
- **OR-Tools**: Google's optimization and constraint programming
- **PySAT**: SAT solvers for Python
- **Fast Downward**: State-of-the-art classical planner

### Tools
- **PDDL**: Planning Domain Definition Language
- **MiniZinc**: Constraint modeling language
- **Z3**: Theorem prover and SMT solver

---

**Congratulations on completing Module 13: Classical AI Algorithms and Search!** 🎉

You now have a comprehensive understanding of the fundamental algorithms that power modern AI systems. From search and game playing to optimization and planning, these techniques form the backbone of intelligent systems.

Keep exploring and building! 🚀
