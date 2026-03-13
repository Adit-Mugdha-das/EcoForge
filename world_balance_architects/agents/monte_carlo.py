# =============================================================================
# agents/monte_carlo.py — Agent 2: Monte Carlo Simulation
# =============================================================================
#
# Decision style: Simulation-based planning (PDF Figure 2)
#
# Algorithm (from PDF):
#   For each candidate action:
#     1. Apply the action to a copy of the world.
#     2. Run `num_rollouts` random simulations (rollouts) from that state.
#     3. Each rollout plays `rollout_depth` half-turns with random-but-valid moves.
#     4. Score the final state with evaluate_state().
#     5. Track the average score across rollouts.
#   Pick the action with the highest average rollout score.
#
# PDF example (Figure 2):
#   Plant Forest → Avg Score 64
#   Build Farm   → Avg Score 72  ← selected
#   Build Canal  → Avg Score 58
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import random
from config import *
from engine.actions import get_all_valid_moves, apply_action
from engine.simulate import simulate
from agents.base_agent import BaseAgent
from agents.eval import evaluate_state


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo simulation agent.

    For each legal move, runs `num_rollouts` random-playout simulations
    of `rollout_depth` half-turns and picks the move with the best average score.
    """

    def __init__(self, agent_id: str, num_rollouts: int = 15,
                 rollout_depth: int = 8, max_candidates: int = 20):
        super().__init__(agent_id)
        self.num_rollouts    = num_rollouts
        self.rollout_depth   = rollout_depth
        self.max_candidates  = max_candidates   # cap first-level branching

    @property
    def name(self) -> str:
        return f"MonteCarlo(rollouts={self.num_rollouts}, depth={self.rollout_depth})"

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def choose_action(self, world) -> tuple:
        """
        Evaluate every candidate action via rollouts and return the best one.
        Falls back to (None, -1, -1) if no valid moves.
        """
        # Pre-select candidates using greedy ordering to avoid simulating
        # obviously bad moves `num_rollouts` times each
        candidates = self.get_sorted_moves(world, self.agent_id, self.max_candidates)
        if not candidates:
            return None, -1, -1

        best_avg   = float('-inf')
        best_move  = candidates[0]

        for action, r, c in candidates:
            # Apply this action once, get the resulting state
            w_after = copy.deepcopy(world)
            apply_action(w_after, self.agent_id, action, r, c)
            simulate(w_after)

            # Run num_rollouts random simulations from that state
            total = 0.0
            for _ in range(self.num_rollouts):
                score  = self._rollout(w_after)
                total += score

            avg = total / self.num_rollouts

            if avg > best_avg:
                best_avg  = avg
                best_move = (action, r, c)

        return best_move

    # -------------------------------------------------------------------------
    # Rollout
    # -------------------------------------------------------------------------

    def _rollout(self, world) -> float:
        """
        Play `rollout_depth` half-turns from `world` using random-but-valid moves
        for both agents, then score the final state from this agent's perspective.

        A half-turn = one agent acts → simulate.
        Both agents alternate randomly, mirroring real game alternation.
        """
        w = copy.deepcopy(world)

        for _ in range(self.rollout_depth):
            over, _ = w.is_game_over()
            if over:
                break

            current = w.current_agent
            moves   = get_all_valid_moves(w, current)

            if moves:
                action, r, c = random.choice(moves)
                apply_action(w, current, action, r, c)

            simulate(w)
            w.switch_agent()

        return evaluate_state(w, self.agent_id)
