# =============================================================================
# agents/q_learning.py — Agent 3: Q-Learning
# =============================================================================
#
# Decision style: Experience-based learning (PDF Figure 3)
#
# Algorithm (from PDF):
#   Maintains a Q-table: State → Action → Value
#   State: tuple of 4 categorical values (water, food, oxygen, temp) → 81 states
#   Actions: 7 action types → Q-table size: 81 × 7 = 567 entries
#
#   Training loop (many episodes before the real game):
#     1. Observe current state S (categorical)
#     2. Select action via epsilon-greedy (explore vs exploit)
#     3. Apply action, simulate, observe next state S'
#     4. Receive reward = environment improves (+) or worsens (-)
#     5. Update Q-table: Q[S,a] += α * (r + γ * max(Q[S']) - Q[S,a])
#     6. Next turn → repeat
#
# After training, epsilon is low so the agent mostly exploits the Q-table.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import copy
import random
from config import *
from engine.world import World
from engine.actions import get_all_valid_moves, apply_action
from engine.simulate import simulate
from agents.base_agent import BaseAgent
from agents.eval import compute_reward, evaluate_state


# Map action names → integer index in Q-table
# All three AdjustAllocation flavours share index 6 (mode detail is secondary)
ACTION_INDEX = {
    'Build Canal':       0,
    'Place Reservoir':   1,
    'Plant Forest':      2,
    'Clear Forest':      3,
    'Plant Crop':        4,
    'Harvest Crop':      5,
    'Adjust Allocation': 6,
}
N_ACTIONS = 7


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent.

    Uses a dictionary Q-table keyed by the abstracted 4-tuple state
    (water_cat, food_cat, oxygen_cat, temp_cat).

    Must be trained before playing via .train().
    Can save/load Q-table to disk so training is not repeated each run.
    """

    def __init__(self, agent_id: str,
                 alpha:   float = ALPHA,
                 gamma:   float = GAMMA,
                 epsilon: float = EPSILON):
        super().__init__(agent_id)
        self.alpha   = alpha    # learning rate
        self.gamma   = gamma    # discount factor
        self.epsilon = epsilon  # current exploration rate (decays during training)

        # Q-table: {state_tuple: [q0, q1, ..., q6]}
        # Initialised to 0.0 (optimistic initialisation could also be used)
        self.q_table: dict = {}
        self.trained  = False

    @property
    def name(self) -> str:
        status = "trained" if self.trained else "untrained"
        return f"QLearning(α={self.alpha}, γ={self.gamma}, {status})"

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def choose_action(self, world) -> tuple:
        """
        Epsilon-greedy action selection.

        With probability epsilon → explore (random valid action).
        Otherwise          → exploit (action with highest Q-value for this state).
        """
        state = world.get_state_category()
        moves = get_all_valid_moves(world, self.agent_id)

        if not moves:
            return None, -1, -1

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(moves)

        # Exploitation: pick move whose action type has best Q-value
        q_vals   = self._get_q_values(state)
        best_val = float('-inf')
        best_move = moves[0]

        for action, r, c in moves:
            idx = ACTION_INDEX.get(action.name, 0)
            if q_vals[idx] > best_val:
                best_val  = q_vals[idx]
                best_move = (action, r, c)

        return best_move

    # -------------------------------------------------------------------------
    # Q-table operations
    # -------------------------------------------------------------------------

    def _get_q_values(self, state: tuple) -> list:
        """Return Q-values for a state; initialise to 0.0 if unseen."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * N_ACTIONS
        return self.q_table[state]

    def update(self, state: tuple, action_idx: int,
               reward: float, next_state: tuple):
        """
        Standard Q-Learning update (Bellman equation):
          Q[s,a] ← Q[s,a] + α * (r + γ * max_a'(Q[s',a']) - Q[s,a])
        """
        q_vals      = self._get_q_values(state)
        next_q_vals = self._get_q_values(next_state)

        td_target = reward + self.gamma * max(next_q_vals)
        td_error  = td_target - q_vals[action_idx]
        q_vals[action_idx] += self.alpha * td_error

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def train(self, episodes: int = TRAIN_EPISODES, verbose: bool = True):
        """
        Pre-train the Q-agent by running `episodes` simulated games.

        Each episode:
          - A fresh World is created.
          - The Q-agent plays its turns using epsilon-greedy.
          - The opponent plays random valid moves.
          - Q-table is updated after every Q-agent action.
          - Epsilon decays each episode (more exploitation over time).

        Reward signal (from PDF): environment improvement or worsening.
        Implemented as: stability_delta * 20 + bonuses/penalties.
        """
        if verbose:
            print(f"[Q-Learning] Training for {episodes} episodes...")

        opponent_id = self.opponent
        eps_start   = self.epsilon
        eps_end     = 0.05
        eps_decay   = (eps_start - eps_end) / max(episodes, 1)

        for ep in range(episodes):
            world = World()

            for _ in range(MAX_TURNS * 2):   # *2 because half-turns
                over, _ = world.is_game_over()
                if over:
                    break

                if world.current_agent == self.agent_id:
                    # ---- Q-agent's half-turn ----
                    state        = world.get_state_category()
                    old_stability = world.stability
                    moves        = get_all_valid_moves(world, self.agent_id)

                    if not moves:
                        simulate(world)
                        world.switch_agent()
                        continue

                    # Epsilon-greedy selection
                    if random.random() < self.epsilon:
                        action, r, c = random.choice(moves)
                    else:
                        q_vals    = self._get_q_values(state)
                        best_idx  = max(range(len(moves)),
                                        key=lambda i: q_vals[ACTION_INDEX.get(moves[i][0].name, 0)])
                        action, r, c = moves[best_idx]

                    action_idx = ACTION_INDEX.get(action.name, 0)

                    # Apply, simulate
                    apply_action(world, self.agent_id, action, r, c)
                    simulate(world)
                    world.switch_agent()

                    # Reward: how much did environment change?
                    reward     = compute_reward(old_stability, world, self.agent_id)
                    next_state = world.get_state_category()

                    # Update Q-table
                    self.update(state, action_idx, reward, next_state)

                else:
                    # ---- Opponent plays randomly ----
                    opp_moves = get_all_valid_moves(world, opponent_id)
                    if opp_moves:
                        a, r, c = random.choice(opp_moves)
                        apply_action(world, opponent_id, a, r, c)
                    simulate(world)
                    world.switch_agent()

            # Decay epsilon linearly toward eps_end
            self.epsilon = max(eps_end, self.epsilon - eps_decay)

        # After training, lock epsilon to exploitation level
        self.epsilon  = eps_end
        self.trained  = True

        if verbose:
            states_learned = len(self.q_table)
            print(f"[Q-Learning] Done. States learned: {states_learned} / 81  "
                  f"(epsilon locked at {self.epsilon})")

    # -------------------------------------------------------------------------
    # Persistence — save and load Q-table
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """Save Q-table to a JSON file so training is not repeated."""
        # Convert tuple keys to strings for JSON serialisation
        serialisable = {str(k): v for k, v in self.q_table.items()}
        with open(path, 'w') as f:
            json.dump({'q_table': serialisable, 'trained': self.trained}, f, indent=2)
        print(f"[Q-Learning] Q-table saved to {path}")

    def load(self, path: str):
        """Load a previously saved Q-table."""
        with open(path, 'r') as f:
            data = json.load(f)
        # Convert string keys back to tuples
        self.q_table = {
            tuple(map(int, k.strip('()').split(','))): v
            for k, v in data['q_table'].items()
        }
        self.trained  = data.get('trained', True)
        self.epsilon  = 0.05   # exploitation mode
        print(f"[Q-Learning] Q-table loaded from {path}  "
              f"(states: {len(self.q_table)})")
