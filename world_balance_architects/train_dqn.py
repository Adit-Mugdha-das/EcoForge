# =============================================================================
# train_dqn.py — Headless DQN pre-training script (no pygame window)
# =============================================================================
#
# Run this ONCE before playing with the DQN agent:
#   python train_dqn.py              # train Agent A, 3000 episodes
#   python train_dqn.py --agent B    # train Agent B
#   python train_dqn.py --episodes 5000
#
# Training strategy — two phases:
#   Phase 1 (first 2/3):  DQN vs Random opponent
#                         Fast — learns basic survival and resource management
#   Phase 2 (last 1/3):   DQN vs Greedy-1-step opponent
#                         Smarter opponent (best immediate eval) without the
#                         deepcopy cost of Monte Carlo — trains competitiveness
#
# Why not Monte Carlo in training?
#   MC does 8 rollouts x 5 deepcopy+simulate per turn — too slow for 3000 eps.
#   Greedy-1-step is ~50x faster and still much harder than random.
#   Competitive adaptation against real MC/Minimax happens via online learning
#   during actual game sessions (main.py updates the model every turn played).
#
# Output: dqn_model_A.pt saved next to this file.
# main.py loads it automatically when you select DQN in the UI.
# =============================================================================

import sys
import os
import random
import copy
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from engine.world import World
from engine.actions import get_all_valid_moves, apply_action
from engine.simulate import simulate
from agents.dqn_agent import DQNAgent, DQN_ACTION_INDEX
from agents.eval import compute_reward, evaluate_state


def _model_path(agent_id: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"dqn_model_{agent_id}.pt")


def _greedy_move(world, opponent_id):
    """
    Pick the opponent move with the best immediate evaluate_state score.
    No deepcopy+simulate — fast enough to use in a training loop.
    """
    moves = get_all_valid_moves(world, opponent_id)
    if not moves:
        return None, -1, -1

    best_score = float('-inf')
    best_move  = moves[0]

    # Score a random sample of 20 moves for speed (not all 100+)
    sample = random.sample(moves, min(len(moves), 20))
    for action, r, c in sample:
        w = copy.copy(world)          # shallow copy — cheap
        try:
            score = evaluate_state(w, opponent_id)
            if score > best_score:
                best_score = score
                best_move  = (action, r, c)
        except Exception:
            pass

    return best_move


def _run_opponent_turn(world, opponent_id, use_greedy: bool):
    """Apply one opponent turn then simulate."""
    if use_greedy:
        a, r, c = _greedy_move(world, opponent_id)
    else:
        moves = get_all_valid_moves(world, opponent_id)
        a, r, c = random.choice(moves) if moves else (None, -1, -1)

    if a is not None:
        try:
            apply_action(world, opponent_id, a, r, c)
        except Exception:
            pass

    simulate(world)
    world.switch_agent()


def run_training(agent_id: str = AGENT_A,
                 episodes:  int = 3000,
                 verbose:   bool = True) -> DQNAgent:
    """
    Train a DQN agent for `episodes` episodes without rendering.
    Returns the trained agent (also saves to dqn_model_{agent_id}.pt).
    """
    opponent_id  = AGENT_B if agent_id == AGENT_A else AGENT_A
    phase2_start = int(episodes * 2 / 3)   # greedy starts at 2/3 mark

    agent     = DQNAgent(agent_id)
    win_count = 0
    recent_rew = []

    if verbose:
        print(f"\n{'='*60}", flush=True)
        print(f"  DQN Training  |  Agent {agent_id}  |  {episodes} episodes",
              flush=True)
        print(f"  Phase 1  ep    1 – {phase2_start:4d}  vs Random opponent",
              flush=True)
        print(f"  Phase 2  ep {phase2_start+1:4d} – {episodes:4d}  vs Greedy-1-step opponent",
              flush=True)
        print(f"{'='*60}", flush=True)

    for ep in range(episodes):
        world      = World()
        use_greedy = ep >= phase2_start
        total_rew  = 0.0

        for _ in range(MAX_TURNS * 2):
            over, _ = world.is_game_over()
            if over:
                break

            if world.current_agent == agent_id:
                # ── DQN agent's turn ──────────────────────────────────────
                state_vec     = agent.get_state_vector(world)
                pre_stability = world.stability
                moves         = get_all_valid_moves(world, agent_id)

                if not moves:
                    simulate(world)
                    world.switch_agent()
                    continue

                action, r, c = agent.choose_action(world)
                action_idx   = DQN_ACTION_INDEX.get(action.name, 0)

                apply_action(world, agent_id, action, r, c)
                simulate(world)
                world.switch_agent()

                over2, _  = world.is_game_over()
                reward    = compute_reward(pre_stability, world, agent_id)
                next_sv   = agent.get_state_vector(world)

                agent.update(state_vec, action_idx, reward, next_sv, done=over2)
                total_rew += reward

            else:
                # ── Opponent's turn ───────────────────────────────────────
                _run_opponent_turn(world, opponent_id, use_greedy)

        # ── Per-episode bookkeeping ───────────────────────────────────────
        if world.get_winner() == agent_id:
            win_count += 1

        recent_rew.append(total_rew)
        if len(recent_rew) > 100:
            recent_rew.pop(0)

        agent.decay_epsilon(ep, episodes)

        if verbose and (ep + 1) % 100 == 0:
            avg_rew  = sum(recent_rew) / len(recent_rew)
            win_rate = win_count / (ep + 1) * 100.0
            phase    = "Greedy" if use_greedy else "Random"
            print(f"  ep {ep+1:4d}/{episodes}"
                  f"  avg_r={avg_rew:+7.1f}"
                  f"  win={win_rate:5.1f}%"
                  f"  e={agent.epsilon:.3f}"
                  f"  buf={len(agent.buffer):5d}"
                  f"  [{phase}]",
                  flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    agent.trained = True
    path = _model_path(agent_id)
    agent.save(path)

    if verbose:
        print(f"\n  Final win rate : {win_count/episodes*100:.1f}%  over {episodes} ep",
              flush=True)
        print(f"  Buffer size    : {len(agent.buffer)} transitions",
              flush=True)
        print(f"  Network updates: {agent.steps}",
              flush=True)
        print(f"\n  Model saved to {path}", flush=True)
        print("  Select DQN in main.py to play with this model.\n", flush=True)

    return agent


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DQN agent headlessly (no pygame window)."
    )
    parser.add_argument('--agent',    default='A', choices=['A', 'B'],
                        help="Agent slot to train (default: A)")
    parser.add_argument('--episodes', type=int, default=3000,
                        help="Training episodes (default: 3000)")
    args = parser.parse_args()

    agent_id = AGENT_A if args.agent == 'A' else AGENT_B
    run_training(agent_id=agent_id, episodes=args.episodes)
