# =============================================================================
# agents/minimax.py — Agent 1: Minimax + Alpha-Beta Pruning
# =============================================================================
#
# Decision style: Adversarial planning (from PDF Figure 1)
#
# Algorithm:
#   Build a state-space tree to `depth` levels.
#   Maximising nodes = this agent's turns (picks the highest-scoring action).
#   Minimising nodes = opponent's turns (picks action that hurts us most).
#   Alpha-beta pruning skips branches that cannot change the outcome.
#
# Performance tuning:
#   `max_branches` caps how many moves are expanded at each node.
#   Moves are pre-sorted by greedy evaluation so the best ones are
#   explored first — this maximises alpha-beta cut-offs.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from config import *
from engine.actions import get_all_valid_moves, apply_action
from engine.simulate import simulate
from agents.base_agent import BaseAgent
from agents.eval import evaluate_state


class MinimaxAgent(BaseAgent):
    """
    Minimax agent with alpha-beta pruning.

    Searches `depth` half-turns ahead (one half-turn = one agent's action).
    depth=2 means: this agent acts → opponent acts → evaluate.
    Branching factor capped at `max_branches` via greedy move ordering.
    """

    def __init__(self, agent_id: str, depth: int = 2, max_branches: int = 12):
        super().__init__(agent_id)
        self.depth       = depth
        self.max_branches = max_branches

    @property
    def name(self) -> str:
        return f"Minimax(depth={self.depth})"

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def choose_action(self, world) -> tuple:
        """
        Return the best (action, row, col) found by minimax search.
        Falls back to (None, -1, -1) if no valid moves exist.
        """
        moves = self.get_sorted_moves(world, self.agent_id, self.max_branches)
        if not moves:
            return None, -1, -1

        best_score = float('-inf')
        best_move  = moves[0]

        alpha = float('-inf')
        beta  = float('inf')

        for action, r, c in moves:
            # Simulate this agent's action
            w_next = self._apply_and_simulate(world, self.agent_id, action, r, c)

            # Now it's the opponent's turn → minimising node
            score = self._minimax(
                w_next,
                depth       = self.depth - 1,
                maximising  = False,
                alpha       = alpha,
                beta        = beta,
            )

            if score > best_score:
                best_score = score
                best_move  = (action, r, c)

            alpha = max(alpha, best_score)
            # No beta pruning at root (we want the full best-move sweep)

        return best_move

    # -------------------------------------------------------------------------
    # Recursive minimax
    # -------------------------------------------------------------------------

    def _minimax(self, world, depth: int, maximising: bool,
                 alpha: float, beta: float) -> float:
        """
        Recursive minimax with alpha-beta pruning.

        maximising=True  → this agent's turn (we want the highest score).
        maximising=False → opponent's turn   (opponent minimises our score).
        """
        # Base cases
        over, _ = world.is_game_over()
        if depth == 0 or over:
            return evaluate_state(world, self.agent_id)

        current_agent = self.agent_id if maximising else self.opponent
        moves = self.get_sorted_moves(world, current_agent, self.max_branches)

        if not moves:
            # No valid moves — evaluate current state
            return evaluate_state(world, self.agent_id)

        if maximising:
            best = float('-inf')
            for action, r, c in moves:
                w_next = self._apply_and_simulate(world, current_agent, action, r, c)
                val    = self._minimax(w_next, depth - 1, False, alpha, beta)
                best   = max(best, val)
                alpha  = max(alpha, best)
                if beta <= alpha:
                    break   # β cut-off: opponent won't allow this branch
            return best

        else:
            best = float('inf')
            for action, r, c in moves:
                w_next = self._apply_and_simulate(world, current_agent, action, r, c)
                val    = self._minimax(w_next, depth - 1, True, alpha, beta)
                best   = min(best, val)
                beta   = min(beta, best)
                if beta <= alpha:
                    break   # α cut-off: we already have a better option
            return best

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _apply_and_simulate(self, world, agent_id: str, action, r: int, c: int):
        """Deep-copy world, apply action, run simulation, return copy."""
        w = copy.deepcopy(world)
        apply_action(w, agent_id, action, r, c)
        simulate(w)
        return w
