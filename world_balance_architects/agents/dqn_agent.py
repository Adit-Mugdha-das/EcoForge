# =============================================================================
# agents/dqn_agent.py — Agent 4: Deep Q-Network (DQN)
# =============================================================================
#
# How DQN differs from Q-Learning:
#   Q-Learning:  lookup table of 243 discrete states → 8 action values
#   DQN:         neural network takes 9 continuous features → 8 action values
#
# Network architecture:
#   Input  (11)  → Hidden (256) → Hidden (256) → Output (8 Q-values)
#
# Key DQN techniques used:
#   1. Experience Replay   — store past transitions, sample random batches
#                            (breaks correlation between consecutive updates)
#   2. Target Network      — a slow-update copy of the network used for
#                            computing TD targets (prevents oscillation)
#   3. Gradient Clipping   — caps gradients at 1.0 (training stability)
#   4. Huber Loss          — less sensitive to outliers than MSE
#
# State vector (11 features, all normalized to [0, 1]):
#   water, food, oxygen, temperature, population,
#   stability, own_eco, own_cells, opp_cells,
#   opp_eco, score_diff
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import *
from agents.base_agent import BaseAgent
from engine.actions import get_all_valid_moves


# ── Constants ────────────────────────────────────────────────────────────────

DQN_ACTION_INDEX = {
    'Plant Forest':               0,
    'Build Canal':                1,
    'Build Reservoir':            2,
    'Plant Farm':                 3,
    'Harvest Crop':               4,
    'Clear Forest':               5,
    'Build Solar Plant':          6,
    'Adjust Resource Allocation': 7,
}
DQN_N_ACTIONS  = 8
DQN_STATE_SIZE = 11   # was 9 — added opp_eco and score_diff features


# =============================================================================
# Neural Network
# =============================================================================

class QNetwork(nn.Module):
    """
    Fully-connected network: 11 inputs → 256 → 256 → 8 Q-values.
    Takes a normalized state vector; outputs one Q-value per action type.
    """
    def __init__(self, state_size: int = DQN_STATE_SIZE,
                 n_actions: int = DQN_N_ACTIONS,
                 hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Experience Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Circular buffer storing (state, action, reward, next_state, done) tuples.
    DQN samples random mini-batches from this buffer to break temporal
    correlation — a key reason DQN is more stable than naive online updates.
    """
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent.

    Maintains two networks (online + target) and a replay buffer.
    Online network is trained every step; target network syncs every
    `target_update` steps to provide stable training targets.

    Can be pre-trained with train_dqn.py (headless, no pygame) and
    then loaded for in-game play.  Also continues learning during play
    via the same update() interface as QLearningAgent.
    """

    def __init__(self, agent_id: str,
                 lr: float            = 0.001,
                 gamma: float         = 0.99,
                 epsilon: float       = 1.0,
                 epsilon_min: float   = 0.05,
                 batch_size: int      = 128,
                 buffer_size: int     = 10_000,
                 target_update: int   = 100):
        super().__init__(agent_id)

        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0
        self.trained       = False

        # Two networks: online (trained every step) + target (updated slowly)
        self.online_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        status = "trained" if self.trained else "untrained"
        return f"DQN({status}, e={self.epsilon:.2f})"

    # ── State representation ──────────────────────────────────────────────────

    def get_state_vector(self, world) -> np.ndarray:
        """
        Build the 11-feature normalized state vector from the current world.
        All values scaled to roughly [0, 1] so the network trains smoothly.

        Features 1-9 are environmental + territorial.
        Features 10-11 are competitive: opponent eco and score differential.
        """
        opponent    = self.opponent
        agent_cells = len(world.get_agent_cells(self.agent_id))
        opp_cells   = len(world.get_agent_cells(opponent))

        # Score differential clipped to [-1, 1]: positive = we're winning
        score_diff = world.scores[self.agent_id] - world.scores[opponent]
        score_diff_norm = max(-1.0, min(1.0, score_diff / 200.0))

        return np.array([
            world.water_level                        / 100.0,
            world.food                               / 100.0,
            world.oxygen                             / 100.0,
            world.temperature                        / 100.0,
            world.population                         / 200.0,
            world.stability,                                   # already 0-1
            world.eco_points[self.agent_id]          / 99.0,
            agent_cells                              / 100.0,
            opp_cells                                / 100.0,
            world.eco_points[opponent]               / 99.0,  # NEW: opp eco spend capacity
            score_diff_norm,                                   # NEW: who is winning
        ], dtype=np.float32)

    # ── Action selection ──────────────────────────────────────────────────────

    def choose_action(self, world) -> tuple:
        """
        Epsilon-greedy selection over valid moves only.
        The eco-hoarding filter is applied FIRST (before the explore/exploit
        split) so eco never piles up even during random exploration.

        Explore: random valid move from the filtered pool.
        Exploit: valid move whose action type has highest Q-value.
        """
        moves = get_all_valid_moves(world, self.agent_id)
        if not moves:
            return None, -1, -1

        # Eco-hoarding filter — runs before epsilon check so it applies to
        # BOTH random exploration and exploitation.
        # When eco > 40, restrict the move pool to actions costing >= 4.
        # Actions that cost < 4 (Adjust=0, Harvest=1, ClearForest=2,
        # Canal/Forest=3) earn less eco than the +2/turn passive income,
        # so eco grows whenever they are chosen. Forcing a >= 4 cost action
        # drains eco back down.
        eco = world.eco_points[self.agent_id]
        if eco > 40:
            expensive_moves = [(a, r, c) for a, r, c in moves if a.cost >= 4]
            if expensive_moves:
                moves = expensive_moves   # restrict pool — cheap actions excluded

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(moves)

        # Exploitation: score each valid move by its Q-value
        state_vec = self.get_state_vector(world)
        with torch.no_grad():
            q_vals = self.online_net(
                torch.FloatTensor(state_vec).unsqueeze(0)
            ).squeeze(0).numpy()

        best_val  = float('-inf')
        best_move = moves[0]
        for action, r, c in moves:
            idx = DQN_ACTION_INDEX.get(action.name, 0)
            if q_vals[idx] > best_val:
                best_val  = q_vals[idx]
                best_move = (action, r, c)

        return best_move

    # ── Learning update ───────────────────────────────────────────────────────

    def update(self, state_vec: np.ndarray, action_idx: int,
               reward: float, next_state_vec: np.ndarray,
               done: bool = False):
        """
        Store transition in replay buffer, then train on a random mini-batch.
        Called after every real game turn (online learning) or training episode turn.

        Bellman target:
          Q(s,a) ← r + γ * max_a' Q_target(s', a')   (if not done)
          Q(s,a) ← r                                   (if done)
        """
        self.buffer.push(state_vec, action_idx, reward, next_state_vec, done)
        self.steps += 1

        # Need at least one full batch before we can train
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        # Q(s, a) from the online network
        q_current = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Target: r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            q_target   = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Huber loss — more robust to large reward outliers than MSE
        loss = nn.SmoothL1Loss()(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # Sync target network every `target_update` steps
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self, episode: int, total_episodes: int):
        """Linear decay from 1.0 to epsilon_min over total_episodes."""
        fraction     = episode / max(total_episodes, 1)
        self.epsilon = max(
            self.epsilon_min,
            1.0 - fraction * (1.0 - self.epsilon_min)
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save network weights + training metadata to a .pt file."""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon':    self.epsilon,
            'steps':      self.steps,
            'trained':    self.trained,
        }, path)
        print(f"[DQN] Model saved to {path}  (steps: {self.steps})")

    def load(self, path: str):
        """Load previously saved weights; switch to near-exploitation mode.
        If the saved model has a different architecture (e.g. old 9-input vs
        new 11-input), raises a clear error asking the user to retrain."""
        data = torch.load(path, weights_only=False)
        try:
            self.online_net.load_state_dict(data['online_net'])
            self.target_net.load_state_dict(data['target_net'])
        except RuntimeError as exc:
            raise RuntimeError(
                f"[DQN] Cannot load '{path}': model architecture changed "
                f"(state size is now {DQN_STATE_SIZE}). "
                "Please retrain: python train_dqn.py"
            ) from exc
        self.epsilon = max(self.epsilon_min, data.get('epsilon', self.epsilon_min))
        self.steps   = data.get('steps', 0)
        self.trained = data.get('trained', True)
        print(f"[DQN] Model loaded from {path}  "
              f"(steps: {self.steps}, e={self.epsilon:.3f})")
