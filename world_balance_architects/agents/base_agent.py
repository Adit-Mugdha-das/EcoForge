# =============================================================================
# agents/base_agent.py — Abstract base class for all AI agents
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
from config import *
from engine.actions import get_all_valid_moves, apply_action
from engine.simulate import simulate
from agents.eval import evaluate_state


class BaseAgent:
    """
    Abstract base for MinimaxAgent, MonteCarloAgent, and QLearningAgent.

    All agents must implement:
      .choose_action(world) → (action, row, col)

    Shared helpers:
      .simulate_action(world, action, row, col) → world_copy
      .get_sorted_moves(world, k)               → top-k moves (for Minimax)
    """

    def __init__(self, agent_id: str):
        assert agent_id in (AGENT_A, AGENT_B), f"Invalid agent_id: {agent_id}"
        self.agent_id = agent_id
        self.opponent = AGENT_B if agent_id == AGENT_A else AGENT_A

    # -------------------------------------------------------------------------
    # Interface (subclasses must implement)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        raise NotImplementedError

    def choose_action(self, world) -> tuple:
        """
        Analyse the world and return the best (action, row, col).
        row and col are -1 for AdjustAllocation actions.
        Returns (None, -1, -1) if no valid move exists.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def simulate_action(self, world, action, row: int, col: int):
        """
        Apply `action` to a deep copy of `world`, run one simulation step,
        and return the resulting world without modifying the original.
        Used by Minimax and Monte Carlo during look-ahead.
        """
        w_copy = copy.deepcopy(world)
        apply_action(w_copy, self.agent_id, action, row, col)
        simulate(w_copy)
        return w_copy

    def get_sorted_moves(self, world, agent_id: str, k: int = 15) -> list:
        """
        Return the top-k valid moves for `agent_id`, sorted by immediate
        greedy evaluation score (best first).

        Capping the branching factor at k is critical for Minimax performance:
        with ~150 valid moves on a 10×10 grid, uncapped depth-2 search would
        require 150² = 22 500 deepcopy+simulate calls per turn.
        With k=15: 15² = 225 — fast enough for real-time play.
        """
        moves = get_all_valid_moves(world, agent_id)
        if not moves or len(moves) <= k:
            return moves

        # Shallow greedy score per move (depth-0 evaluation)
        def quick_score(move):
            action, r, c = move
            w = copy.deepcopy(world)
            apply_action(w, agent_id, action, r, c)
            simulate(w)
            return evaluate_state(w, self.agent_id)

        moves.sort(key=quick_score, reverse=True)
        return moves[:k]

    def __repr__(self) -> str:
        return f"<{self.name} agent_id={self.agent_id}>"
