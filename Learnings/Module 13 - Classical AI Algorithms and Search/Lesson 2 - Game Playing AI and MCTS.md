# Lesson 2: Game Playing AI & MCTS 🎮

**Module 13: Classical AI Algorithms and Search | Lesson 2 of 5**

Master the algorithms that power chess engines, AlphaGo, and modern game AI!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Implement minimax algorithm with alpha-beta pruning
2. ✅ Design evaluation functions for game positions
3. ✅ Understand Monte Carlo Tree Search (MCTS)
4. ✅ Learn how AlphaGo defeated Lee Sedol
5. ✅ Build game-playing agents from scratch
6. ✅ Apply these techniques to chess, Go, and Tic-Tac-Toe

---

## 1. Introduction to Game Playing

### Two-Player Zero-Sum Games

In game theory, a **zero-sum game** means one player's gain is another's loss.

**Key Concepts:**
- **Perfect information**: Both players see the full game state (chess, Go)
- **Deterministic**: No randomness (unlike poker or backgammon)
- **Adversarial search**: Opponent tries to minimize your score

```python
import numpy as np
import math
from typing import List, Tuple, Optional
from copy import deepcopy
import random

class GameState:
    """
    Abstract base class for game states.

    All games must implement this interface.
    """
    def get_legal_actions(self):
        """Return list of legal actions from this state."""
        raise NotImplementedError

    def generate_successor(self, action):
        """Return new state after taking action."""
        raise NotImplementedError

    def is_terminal(self):
        """Check if game is over."""
        raise NotImplementedError

    def get_utility(self, player):
        """
        Get utility value for player.

        Returns:
            +1 if player won
            -1 if player lost
            0 if draw
        """
        raise NotImplementedError

    def get_current_player(self):
        """Return whose turn it is."""
        raise NotImplementedError


class TicTacToe(GameState):
    """
    Tic-Tac-Toe game implementation.

    Board positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """
    def __init__(self, board=None, current_player=1):
        if board is None:
            self.board = [0] * 9  # 0=empty, 1=X, -1=O
        else:
            self.board = board[:]
        self.current_player = current_player

    def get_legal_actions(self):
        """Return indices of empty cells."""
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def generate_successor(self, action):
        """Place mark at position."""
        new_board = self.board[:]
        new_board[action] = self.current_player
        return TicTacToe(new_board, -self.current_player)

    def is_terminal(self):
        """Check if game is over."""
        return self._check_winner() is not None or len(self.get_legal_actions()) == 0

    def get_utility(self, player):
        """Get utility for player."""
        winner = self._check_winner()
        if winner == player:
            return 1
        elif winner == -player:
            return -1
        return 0

    def get_current_player(self):
        return self.current_player

    def _check_winner(self):
        """Check if there's a winner."""
        # Winning combinations
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]

        for line in lines:
            if (self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0):
                return self.board[line[0]]

        return None  # No winner

    def __repr__(self):
        """Pretty print board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for i in range(0, 9, 3):
            row = ' | '.join(symbols[self.board[j]] for j in range(i, i+3))
            rows.append(row)
        return '\n---------\n'.join(rows)


# Test Tic-Tac-Toe
game = TicTacToe()
print("Initial Board:")
print(game)
print(f"\nLegal actions: {game.get_legal_actions()}")

# Make a move
game = game.generate_successor(4)  # X plays center
print(f"\nAfter X plays center:")
print(game)
```

---

## 2. Minimax Algorithm

### The Core Idea

**Minimax** assumes both players play optimally:
- **MAX player**: Maximizes score
- **MIN player**: Minimizes score (opponent)

We recursively evaluate game tree to find best move.

```python
def minimax(state: GameState, depth: int, maximizing_player: bool):
    """
    Minimax algorithm for two-player games.

    Args:
        state: Current game state
        depth: Maximum search depth
        maximizing_player: True if MAX's turn, False if MIN's turn

    Returns:
        (best_value, best_action)
    """
    # Base cases
    if depth == 0 or state.is_terminal():
        # Evaluate from MAX's perspective
        player = 1 if maximizing_player else -1
        return state.get_utility(player), None

    legal_actions = state.get_legal_actions()

    if maximizing_player:
        # MAX player: maximize value
        max_value = -math.inf
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            value, _ = minimax(successor, depth - 1, False)

            if value > max_value:
                max_value = value
                best_action = action

        return max_value, best_action

    else:
        # MIN player: minimize value
        min_value = math.inf
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            value, _ = minimax(successor, depth - 1, True)

            if value < min_value:
                min_value = value
                best_action = action

        return min_value, best_action


class MinimaxAgent:
    """Agent that uses minimax to play games."""

    def __init__(self, depth=9):
        self.depth = depth

    def get_action(self, state: GameState):
        """Get best action using minimax."""
        maximizing = (state.get_current_player() == 1)
        value, action = minimax(state, self.depth, maximizing)
        return action


# Test minimax on Tic-Tac-Toe
print("\n" + "="*60)
print("Minimax Agent Playing Tic-Tac-Toe")
print("="*60)

game = TicTacToe()
agent = MinimaxAgent(depth=9)

move_count = 0
while not game.is_terminal():
    print(f"\nMove {move_count + 1}:")
    print(game)

    action = agent.get_action(game)
    print(f"Player {game.get_current_player()} plays position {action}")

    game = game.generate_successor(action)
    move_count += 1

print("\nFinal Board:")
print(game)
winner = game._check_winner()
if winner:
    print(f"\nWinner: {'X' if winner == 1 else 'O'}")
else:
    print("\nDraw!")
```

**Minimax Properties:**
- ✅ Complete: Always finds optimal move
- ✅ Optimal: Assumes perfect play from both sides
- ❌ Time: O(b^m) where b=branching factor, m=max depth
- ❌ Space: O(bm) for depth-first search

---

## 3. Alpha-Beta Pruning

### Optimization: Prune Useless Branches

Alpha-beta pruning eliminates branches that can't affect the final decision.

**Key Insight:** If we already found a move that's better than what the opponent will allow, we can stop searching!

```python
def alpha_beta_search(state: GameState, depth: int, alpha: float, beta: float,
                      maximizing_player: bool):
    """
    Minimax with alpha-beta pruning.

    Args:
        alpha: Best value for MAX so far
        beta: Best value for MIN so far

    Prunes branches when alpha >= beta.
    """
    # Base cases
    if depth == 0 or state.is_terminal():
        player = 1 if maximizing_player else -1
        return state.get_utility(player), None

    legal_actions = state.get_legal_actions()

    if maximizing_player:
        max_value = -math.inf
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_search(successor, depth - 1, alpha, beta, False)

            if value > max_value:
                max_value = value
                best_action = action

            alpha = max(alpha, value)

            # Beta cutoff: MIN won't allow this branch
            if beta <= alpha:
                break  # Prune remaining actions

        return max_value, best_action

    else:
        min_value = math.inf
        best_action = None

        for action in legal_actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_search(successor, depth - 1, alpha, beta, True)

            if value < min_value:
                min_value = value
                best_action = action

            beta = min(beta, value)

            # Alpha cutoff: MAX won't allow this branch
            if beta <= alpha:
                break  # Prune remaining actions

        return min_value, best_action


class AlphaBetaAgent:
    """Agent using alpha-beta pruning."""

    def __init__(self, depth=9):
        self.depth = depth
        self.nodes_explored = 0

    def get_action(self, state: GameState):
        """Get best action using alpha-beta."""
        self.nodes_explored = 0
        maximizing = (state.get_current_player() == 1)

        value, action = alpha_beta_search(
            state, self.depth,
            -math.inf, math.inf,
            maximizing
        )
        return action


# Compare minimax vs alpha-beta
print("\n" + "="*60)
print("Minimax vs Alpha-Beta Comparison")
print("="*60)

# Count nodes expanded
def count_nodes_minimax(state, depth, maximizing):
    """Modified minimax that counts nodes."""
    count = 1

    if depth == 0 or state.is_terminal():
        return 0, count

    for action in state.get_legal_actions():
        successor = state.generate_successor(action)
        _, nodes = count_nodes_minimax(successor, depth - 1, not maximizing)
        count += nodes

    return None, count


def count_nodes_alpha_beta(state, depth, alpha, beta, maximizing):
    """Modified alpha-beta that counts nodes."""
    count = 1

    if depth == 0 or state.is_terminal():
        return 0, count

    for action in state.get_legal_actions():
        successor = state.generate_successor(action)
        _, nodes = count_nodes_alpha_beta(
            successor, depth - 1, alpha, beta, not maximizing
        )
        count += nodes

        # Pruning logic
        if maximizing:
            alpha = max(alpha, _) if _ is not None else alpha
        else:
            beta = min(beta, _) if _ is not None else beta

        if beta <= alpha:
            break

    return None, count


game = TicTacToe()
_, minimax_nodes = count_nodes_minimax(game, 9, True)
_, alphabeta_nodes = count_nodes_alpha_beta(game, 9, -math.inf, math.inf, True)

print(f"Minimax nodes explored: {minimax_nodes}")
print(f"Alpha-Beta nodes explored: {alphabeta_nodes}")
print(f"Pruning efficiency: {(1 - alphabeta_nodes/minimax_nodes)*100:.1f}% reduction")
```

**Alpha-Beta Properties:**
- ✅ Same optimal result as minimax
- ✅ Much faster: O(b^(m/2)) in best case
- ✅ Can search twice as deep in same time!
- 📊 Move ordering affects pruning efficiency

---

## 4. Evaluation Functions

### Handling Deep Game Trees

For complex games (chess, Go), we can't search to end. We need **evaluation functions**.

```python
class Connect4(GameState):
    """
    Connect Four game.

    7 columns, 6 rows. Drop pieces from top.
    First to connect 4 horizontally/vertically/diagonally wins.
    """
    def __init__(self, board=None, current_player=1):
        if board is None:
            self.board = [[0]*7 for _ in range(6)]
        else:
            self.board = [row[:] for row in board]
        self.current_player = current_player
        self.rows = 6
        self.cols = 7

    def get_legal_actions(self):
        """Return columns that aren't full."""
        return [col for col in range(self.cols) if self.board[0][col] == 0]

    def generate_successor(self, col):
        """Drop piece in column."""
        new_board = [row[:] for row in self.board]

        # Find lowest empty row
        for row in range(self.rows - 1, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = self.current_player
                break

        return Connect4(new_board, -self.current_player)

    def is_terminal(self):
        """Check if game over."""
        return self._check_winner() is not None or len(self.get_legal_actions()) == 0

    def get_utility(self, player):
        """Get utility for player."""
        winner = self._check_winner()
        if winner == player:
            return 1
        elif winner == -player:
            return -1
        return 0

    def get_current_player(self):
        return self.current_player

    def _check_winner(self):
        """Check for 4 in a row."""
        # Check horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row][col+1] ==
                    self.board[row][col+2] == self.board[row][col+3]):
                    return self.board[row][col]

        # Check vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row+1][col] ==
                    self.board[row+2][col] == self.board[row+3][col]):
                    return self.board[row][col]

        # Check diagonal (down-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row+1][col+1] ==
                    self.board[row+2][col+2] == self.board[row+3][col+3]):
                    return self.board[row][col]

        # Check diagonal (down-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if (self.board[row][col] != 0 and
                    self.board[row][col] == self.board[row+1][col-1] ==
                    self.board[row+2][col-2] == self.board[row+3][col-3]):
                    return self.board[row][col]

        return None


def evaluate_connect4(state: Connect4, player: int):
    """
    Evaluation function for Connect Four.

    Features:
    - Count 2-in-a-row, 3-in-a-row (potential 4s)
    - Center column control
    - Blocking opponent's threats
    """
    if state.is_terminal():
        return state.get_utility(player) * 1000

    score = 0

    # Center column preference
    center_col = state.cols // 2
    center_count = sum(1 for row in range(state.rows)
                      if state.board[row][center_col] == player)
    score += center_count * 3

    # Count potential winning positions
    score += count_sequences(state, player, 2) * 2
    score += count_sequences(state, player, 3) * 5

    # Penalize opponent's potential
    score -= count_sequences(state, -player, 2) * 2
    score -= count_sequences(state, -player, 3) * 5

    return score


def count_sequences(state: Connect4, player: int, length: int):
    """Count sequences of 'length' pieces for player."""
    count = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # H, V, Diag

    for row in range(state.rows):
        for col in range(state.cols):
            for dr, dc in directions:
                seq_count = 0
                empty_count = 0

                for i in range(4):
                    r, c = row + i*dr, col + i*dc
                    if not (0 <= r < state.rows and 0 <= c < state.cols):
                        break

                    if state.board[r][c] == player:
                        seq_count += 1
                    elif state.board[r][c] == 0:
                        empty_count += 1
                    else:
                        break

                if seq_count == length and empty_count == 4 - length:
                    count += 1

    return count


class EvaluationAgent:
    """Agent using evaluation function with limited depth."""

    def __init__(self, depth=4, eval_fn=None):
        self.depth = depth
        self.eval_fn = eval_fn

    def get_action(self, state: GameState):
        """Get best action using minimax with evaluation."""
        value, action = self._minimax_eval(
            state, self.depth,
            -math.inf, math.inf,
            True
        )
        return action

    def _minimax_eval(self, state, depth, alpha, beta, maximizing):
        """Minimax with evaluation function."""
        if depth == 0 or state.is_terminal():
            player = state.get_current_player() if maximizing else -state.get_current_player()
            if state.is_terminal():
                return state.get_utility(player) * 1000, None
            return self.eval_fn(state, player), None

        legal_actions = state.get_legal_actions()

        if maximizing:
            max_value = -math.inf
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self._minimax_eval(successor, depth-1, alpha, beta, False)

                if value > max_value:
                    max_value = value
                    best_action = action

                alpha = max(alpha, value)
                if beta <= alpha:
                    break

            return max_value, best_action
        else:
            min_value = math.inf
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self._minimax_eval(successor, depth-1, alpha, beta, True)

                if value < min_value:
                    min_value = value
                    best_action = action

                beta = min(beta, value)
                if beta <= alpha:
                    break

            return min_value, best_action


# Test evaluation-based agent
print("\n" + "="*60)
print("Connect Four with Evaluation Function")
print("="*60)

game = Connect4()
agent = EvaluationAgent(depth=5, eval_fn=evaluate_connect4)

for move in range(5):
    action = agent.get_action(game)
    print(f"Move {move+1}: Player {game.get_current_player()} plays column {action}")
    game = game.generate_successor(action)

    # Display board
    for row in game.board:
        print(' '.join('X' if c == 1 else 'O' if c == -1 else '.' for c in row))
    print()
```

---

## 5. Monte Carlo Tree Search (MCTS)

### The Revolution: AlphaGo's Secret Weapon

MCTS combines:
- **Tree search**: Build game tree incrementally
- **Random simulation**: Estimate position value via random playouts
- **Upper Confidence Bound**: Balance exploration/exploitation

### The Four Phases

1. **Selection**: Traverse tree using UCB
2. **Expansion**: Add new node
3. **Simulation**: Random playout to end
4. **Backpropagation**: Update statistics

```python
class MCTSNode:
    """
    Node in MCTS tree.

    Tracks:
    - visits: Number of times visited
    - value: Cumulative reward
    - children: Child nodes
    """
    def __init__(self, state: GameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action

        self.children = []
        self.visits = 0
        self.value = 0.0

        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self):
        """Check if this is a terminal state."""
        return self.state.is_terminal()

    def best_child(self, c_param=1.4):
        """
        Select best child using UCB1.

        UCB1 = Q(child) + c * sqrt(ln(N(parent)) / N(child))

        - Q(child): Average value
        - N: Visit count
        - c: Exploration parameter
        """
        choices_weights = [
            (child.value / child.visits) +
            c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        """Add a new child node for an untried action."""
        action = self.untried_actions.pop()
        next_state = self.state.generate_successor(action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        """Update node statistics."""
        self.visits += 1
        self.value += result


def mcts_search(root_state: GameState, num_simulations=1000):
    """
    Monte Carlo Tree Search.

    Args:
        root_state: Current game state
        num_simulations: Number of MCTS iterations

    Returns:
        best_action: Most visited action
    """
    root_node = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root_node

        # 1. Selection: Traverse tree using UCB
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # 2. Expansion: Add new node
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation: Random playout
        result = simulate_random_playout(node.state)

        # 4. Backpropagation: Update statistics
        while node is not None:
            # Flip result for alternating players
            player_result = result if node.state.get_current_player() == 1 else -result
            node.update(player_result)
            node = node.parent

    # Return action with most visits
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.action


def simulate_random_playout(state: GameState):
    """
    Simulate random game from state to end.

    Returns:
        utility for player 1
    """
    current_state = state

    while not current_state.is_terminal():
        actions = current_state.get_legal_actions()
        action = random.choice(actions)
        current_state = current_state.generate_successor(action)

    return current_state.get_utility(1)


class MCTSAgent:
    """Agent using Monte Carlo Tree Search."""

    def __init__(self, num_simulations=1000):
        self.num_simulations = num_simulations

    def get_action(self, state: GameState):
        """Get best action using MCTS."""
        return mcts_search(state, self.num_simulations)


# Test MCTS on Tic-Tac-Toe
print("\n" + "="*60)
print("MCTS Agent Playing Tic-Tac-Toe")
print("="*60)

game = TicTacToe()
mcts_agent = MCTSAgent(num_simulations=500)
random_agent = lambda state: random.choice(state.get_legal_actions())

move_count = 0
while not game.is_terminal():
    if game.get_current_player() == 1:
        # MCTS plays as X
        action = mcts_agent.get_action(game)
        print(f"\nMCTS (X) plays position {action}")
    else:
        # Random plays as O
        action = random_agent(game)
        print(f"Random (O) plays position {action}")

    game = game.generate_successor(action)
    print(game)

winner = game._check_winner()
if winner == 1:
    print("\n✅ MCTS (X) wins!")
elif winner == -1:
    print("\n❌ Random (O) wins!")
else:
    print("\nDraw!")
```

**MCTS Properties:**
- ✅ No evaluation function needed
- ✅ Works for huge state spaces (Go)
- ✅ Anytime algorithm (better with more time)
- ✅ Asymmetric tree growth (focuses on promising lines)
- 📊 Performance improves with more simulations

---

## 6. AlphaGo: MCTS + Deep Learning

### How AlphaGo Defeated Lee Sedol

AlphaGo combined:
1. **MCTS**: Tree search framework
2. **Policy Network**: Guide tree search (learned from human games)
3. **Value Network**: Evaluate positions (replace random playouts)
4. **Reinforcement Learning**: Self-play to improve (see Module 12)

```python
class AlphaGoSimplified:
    """
    Simplified AlphaGo-style MCTS.

    Uses neural network (simulated here) to:
    - Guide action selection (policy)
    - Evaluate positions (value)
    """
    def __init__(self, num_simulations=500):
        self.num_simulations = num_simulations

    def policy_network(self, state):
        """
        Simulate policy network.

        In real AlphaGo: Deep CNN trained on expert games.
        Here: Simple heuristic for demo.
        """
        actions = state.get_legal_actions()

        # Simplified: Uniform distribution
        probs = [1.0 / len(actions) for _ in actions]
        return dict(zip(actions, probs))

    def value_network(self, state):
        """
        Simulate value network.

        In real AlphaGo: Deep CNN trained via self-play.
        Here: Random playout for demo.
        """
        # In practice: Return value in [-1, 1]
        return simulate_random_playout(state)

    def get_action(self, state):
        """MCTS guided by policy and value networks."""
        root = MCTSNode(state)

        for _ in range(self.num_simulations):
            node = root

            # Selection with policy network
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(c_param=1.4)

            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Evaluation: Mix of value network + playout
            if node.is_terminal():
                result = node.state.get_utility(1)
            else:
                # 50% value network, 50% random playout
                value_eval = self.value_network(node.state)
                result = value_eval

            # Backpropagation
            while node is not None:
                player_result = result if node.state.get_current_player() == 1 else -result
                node.update(player_result)
                node = node.parent

        # Select most visited action
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action


# Compare MCTS vs AlphaGo-style
print("\n" + "="*60)
print("MCTS vs AlphaGo-Style MCTS")
print("="*60)

wins = {'mcts': 0, 'alphago': 0, 'draw': 0}

for game_num in range(10):
    game = TicTacToe()
    mcts_agent = MCTSAgent(num_simulations=200)
    alphago_agent = AlphaGoSimplified(num_simulations=200)

    while not game.is_terminal():
        if game.get_current_player() == 1:
            action = mcts_agent.get_action(game)
        else:
            action = alphago_agent.get_action(game)

        game = game.generate_successor(action)

    winner = game._check_winner()
    if winner == 1:
        wins['mcts'] += 1
    elif winner == -1:
        wins['alphago'] += 1
    else:
        wins['draw'] += 1

print(f"Results over 10 games:")
print(f"  MCTS wins: {wins['mcts']}")
print(f"  AlphaGo-style wins: {wins['alphago']}")
print(f"  Draws: {wins['draw']}")
```

### AlphaGo Zero: No Human Knowledge

AlphaGo Zero learned entirely from self-play:
- No human game data
- Stronger than original AlphaGo
- Discovered new strategies

**Connection to Module 12 (Reinforcement Learning):**
- AlphaGo used **policy gradient methods** (Lesson 4)
- Self-play is a form of **multi-agent RL** (Lesson 10)
- Value network trained via **TD learning** (Lesson 2)

---

## 7. Advanced Techniques

### Iterative Deepening

```python
def iterative_deepening_minimax(state: GameState, max_depth=10, time_limit=5.0):
    """
    Iterative deepening: Gradually increase search depth.

    Benefits:
    - Get best move found so far if time runs out
    - Better move ordering for alpha-beta
    """
    import time

    start_time = time.time()
    best_action = None

    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break

        value, action = alpha_beta_search(
            state, depth, -math.inf, math.inf,
            state.get_current_player() == 1
        )

        best_action = action

        print(f"Depth {depth}: Best action = {action}, Value = {value:.2f}")

    return best_action


# Test iterative deepening
game = Connect4()
print("\nIterative Deepening on Connect Four:")
action = iterative_deepening_minimax(game, max_depth=6, time_limit=2.0)
print(f"\nChosen action: {action}")
```

### Move Ordering

```python
def order_moves(state: GameState, actions: List, eval_fn):
    """
    Order moves by evaluation function.

    Better move ordering → more alpha-beta pruning!
    """
    move_scores = []

    for action in actions:
        successor = state.generate_successor(action)
        score = eval_fn(successor, state.get_current_player())
        move_scores.append((action, score))

    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)

    return [action for action, _ in move_scores]


def alpha_beta_with_ordering(state, depth, alpha, beta, maximizing, eval_fn):
    """Alpha-beta with move ordering."""
    if depth == 0 or state.is_terminal():
        if state.is_terminal():
            player = state.get_current_player()
            return state.get_utility(player) * 1000, None
        return eval_fn(state, state.get_current_player()), None

    actions = state.get_legal_actions()

    # Order moves by evaluation
    if len(actions) > 1:
        actions = order_moves(state, actions, eval_fn)

    if maximizing:
        max_value = -math.inf
        best_action = None

        for action in actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_with_ordering(
                successor, depth-1, alpha, beta, False, eval_fn
            )

            if value > max_value:
                max_value = value
                best_action = action

            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return max_value, best_action
    else:
        min_value = math.inf
        best_action = None

        for action in actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_with_ordering(
                successor, depth-1, alpha, beta, True, eval_fn
            )

            if value < min_value:
                min_value = value
                best_action = action

            beta = min(beta, value)
            if beta <= alpha:
                break

        return min_value, best_action
```

### Transposition Tables

```python
class TranspositionTable:
    """
    Cache for game positions.

    Store computed values to avoid re-evaluating same position.
    """
    def __init__(self):
        self.table = {}

    def get(self, state_hash):
        """Lookup cached value."""
        return self.table.get(state_hash)

    def store(self, state_hash, depth, value, action):
        """Store evaluated position."""
        if state_hash not in self.table or self.table[state_hash]['depth'] < depth:
            self.table[state_hash] = {
                'depth': depth,
                'value': value,
                'action': action
            }

    def size(self):
        """Number of cached positions."""
        return len(self.table)


def state_hash(state):
    """Create hash for game state."""
    # For TicTacToe/Connect4: Hash board configuration
    if hasattr(state, 'board'):
        if isinstance(state.board, list):
            if isinstance(state.board[0], list):
                # 2D board (Connect4)
                return hash(tuple(tuple(row) for row in state.board))
            else:
                # 1D board (TicTacToe)
                return hash(tuple(state.board))
    return hash(str(state))


def alpha_beta_with_tt(state, depth, alpha, beta, maximizing, tt):
    """Alpha-beta with transposition table."""
    s_hash = state_hash(state)

    # Check transposition table
    cached = tt.get(s_hash)
    if cached and cached['depth'] >= depth:
        return cached['value'], cached['action']

    # Standard alpha-beta logic
    if depth == 0 or state.is_terminal():
        player = state.get_current_player()
        value = state.get_utility(player) if state.is_terminal() else 0
        tt.store(s_hash, depth, value, None)
        return value, None

    actions = state.get_legal_actions()
    best_action = None

    if maximizing:
        max_value = -math.inf

        for action in actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_with_tt(successor, depth-1, alpha, beta, False, tt)

            if value > max_value:
                max_value = value
                best_action = action

            alpha = max(alpha, value)
            if beta <= alpha:
                break

        tt.store(s_hash, depth, max_value, best_action)
        return max_value, best_action
    else:
        min_value = math.inf

        for action in actions:
            successor = state.generate_successor(action)
            value, _ = alpha_beta_with_tt(successor, depth-1, alpha, beta, True, tt)

            if value < min_value:
                min_value = value
                best_action = action

            beta = min(beta, value)
            if beta <= alpha:
                break

        tt.store(s_hash, depth, min_value, best_action)
        return min_value, best_action


# Test with transposition table
tt = TranspositionTable()
game = TicTacToe()

value, action = alpha_beta_with_tt(game, 9, -math.inf, math.inf, True, tt)
print(f"\nTransposition table size: {tt.size()} positions cached")
```

---

## 8. Practice Exercises

### Exercise 1: Implement Negamax

```python
"""
Negamax is a simplified version of minimax.

Key insight: max(a, b) = -min(-a, -b)

Implement negamax with alpha-beta pruning.
Single function instead of separate max/min logic.
"""

# Your implementation here
```

### Exercise 2: Chess Evaluation Function

```python
"""
Design an evaluation function for chess.

Consider:
- Material: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9
- Position: Center control, king safety
- Mobility: Number of legal moves
- Pawn structure: Doubled pawns, isolated pawns

Bonus: Implement minimax chess AI!
"""

# Your implementation here
```

### Exercise 3: UCT for MCTS

```python
"""
Implement UCT (Upper Confidence bounds for Trees) variant of MCTS.

Enhancements:
- Progressive widening: Don't expand all children immediately
- RAVE (Rapid Action Value Estimation): Share info across siblings
- Virtual loss: Parallel MCTS

Test on Connect Four or Go (small board).
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Minimax Algorithm** 🎯
   - Assumes optimal play from both players
   - Recursively evaluates game tree
   - Complete and optimal for perfect information games
   - O(b^m) time complexity

2. **Alpha-Beta Pruning** ✂️
   - Eliminates branches that can't affect decision
   - Same result as minimax, much faster
   - O(b^(m/2)) in best case
   - Move ordering crucial for efficiency

3. **Evaluation Functions** 📊
   - Estimate position value for non-terminal states
   - Combine multiple features (material, position, mobility)
   - Quality determines playing strength
   - Domain knowledge matters

4. **Monte Carlo Tree Search** 🎲
   - No evaluation function needed
   - Random playouts estimate position value
   - UCB balances exploration/exploitation
   - Powers AlphaGo and modern game AI

5. **AlphaGo Revolution** 🏆
   - MCTS + deep learning
   - Policy network guides search
   - Value network evaluates positions
   - Self-play for superhuman performance

### Algorithm Comparison

| Algorithm | Evaluation | Search Depth | Best For |
|-----------|-----------|--------------|----------|
| Minimax | Required | Limited | Simple games |
| Alpha-Beta | Required | 2x deeper | Chess, checkers |
| MCTS | Not needed | Adaptive | Go, complex games |
| AlphaGo | Neural nets | Deep + selective | Any game |

### Real-World Applications

✅ **Chess Engines**: Stockfish uses alpha-beta + evaluation
✅ **Go AI**: AlphaGo, KataGo use MCTS + neural networks
✅ **Game AI**: Strategy games, puzzle solving
✅ **Planning**: Sequential decision-making
✅ **Robotics**: Adversarial scenarios

### What's Next?

In Lesson 3, we'll explore **Optimization Algorithms**:
- Genetic algorithms for search
- Simulated annealing
- Particle swarm optimization
- Applications to hyperparameter tuning

**Game AI shows the power of tree search!** 🚀

---

## Additional Resources

### Papers
- Shannon (1950): "Programming a Computer for Playing Chess"
- Campbell et al. (2002): "Deep Blue" (IBM chess computer)
- Silver et al. (2016): "Mastering Go with Deep Neural Networks and Tree Search" (AlphaGo)
- Silver et al. (2017): "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero)

### Books
- **Artificial Intelligence: A Modern Approach** (Russell & Norvig) - Chapter 5
- **Deep Learning and the Game of Go** (Burchill & DeVries)

### Code & Tools
- **python-chess**: Chess library with AI
- **AlphaZero General**: AlphaZero implementation for any game
- **OpenSpiel**: Multi-agent RL and game playing

### Visualizations
- **Chess Programming Wiki**: Alpha-beta visualizations
- **MCTS Visualizer**: Interactive MCTS demo

---

**Next**: [Lesson 3 - Optimization Algorithms](Lesson%203%20-%20Optimization%20Algorithms.md)

Discover evolutionary and swarm-based optimization! 🧬
