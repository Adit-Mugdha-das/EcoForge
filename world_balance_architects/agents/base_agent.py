# =============================================================================
# agents/base_agent.py — Abstract base class for all AI agents
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import random
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
        Return the top-k valid moves, sorted by a cheap greedy score.

        Performance fixes vs naive approach:
        - If >k*3 valid moves exist, randomly sample k*3 before scoring.
          (avoids scoring all 150+ moves at every minimax node)
        - quick_score does NOT call simulate() — just apply + evaluate.
          (simulate is expensive; skipping it makes sorting ~10x faster)
        - try/except guards against any edge-case crash in evaluation.
        """
        moves = get_all_valid_moves(world, agent_id)
        if not moves:
            return []
        if len(moves) <= k:
            return moves

        # Random pre-sample: score at most k*3 candidates, not all 150+
        pool = random.sample(moves, min(len(moves), k * 3))

        def quick_score(move):
            try:
                action, r, c = move
                w = copy.deepcopy(world)
                apply_action(w, agent_id, action, r, c)
                # No simulate() here — apply-only evaluation is fast enough
                # for move ordering, and simulate is the expensive part
                return evaluate_state(w, self.agent_id)
            except Exception:
                return float('-inf')   # failed = treat as worst option

        pool.sort(key=quick_score, reverse=True)
        return pool[:k]

    def __repr__(self) -> str:
        return f"<{self.name} agent_id={self.agent_id}>"
