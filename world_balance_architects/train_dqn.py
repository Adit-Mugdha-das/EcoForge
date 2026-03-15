# =============================================================================
# train_dqn.py — Headless DQN pre-training script (no pygame window)
# =============================================================================
#
# Run this ONCE before playing with the DQN agent:
#   python train_dqn.py              # train Agent A, 5000 episodes
#   python train_dqn.py --agent B    # train Agent B
#   python train_dqn.py --episodes 8000
#
# Training strategy — four phases:
#   Phase 1 (first 40%):  DQN vs Random opponent
#                         Learns basic survival, resource management, building.
#   Phase 2 (next 30%):   DQN vs Greedy-1-step opponent (5 candidates)
#                         Weak greedy — moderate pressure.
#   Phase 3 (next 20%):   DQN vs Greedy-1-step opponent (10 candidates)
#                         Medium greedy — stronger competitive play.
#   Phase 4 (last 10%):   DQN vs Greedy-1-step opponent (15 candidates)
#                         Strong greedy — close to depth-1 best-move pressure.
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
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"dqn_model_{agent_id}.pt"
    )


def _greedy_move(world, opponent_id, candidates: int = 15):
    """
    Pick the opponent move with the best immediate evaluate_state score.
    deepcopy → apply_action → simulate → score (correct depth-1 search).

    candidates=5   -> weak greedy
    candidates=10  -> medium greedy
    candidates=15  -> strong greedy
    """
    moves = get_all_valid_moves(world, opponent_id)
    if not moves:
        return None, -1, -1

    best_score = float('-inf')
    best_move = moves[0]

    sample = random.sample(moves, min(len(moves), candidates))
    for action, r, c in sample:
        w = copy.deepcopy(world)
        try:
            apply_action(w, opponent_id, action, r, c)
            simulate(w)
            score = evaluate_state(w, opponent_id)
            if score > best_score:
                best_score = score
                best_move = (action, r, c)
        except Exception:
            pass

    return best_move


def _run_opponent_turn(world, opponent_id, greedy_candidates: int = 0):
    """
    Apply one opponent turn then simulate.

    greedy_candidates=0   -> random
    greedy_candidates=5   -> weak greedy
    greedy_candidates=10  -> medium greedy
    greedy_candidates=15  -> strong greedy
    """
    if greedy_candidates > 0:
        a, r, c = _greedy_move(world, opponent_id, candidates=greedy_candidates)
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
                 episodes: int = 5000,
                 verbose: bool = True) -> DQNAgent:
    """
    Train a DQN agent for `episodes` episodes without rendering.

    Four-phase curriculum:
    Phase 1 (40%): Random
    Phase 2 (30%): Greedy-5
    Phase 3 (20%): Greedy-10
    Phase 4 (10%): Greedy-15
    """
    opponent_id = AGENT_B if agent_id == AGENT_A else AGENT_A

    phase2_start = int(episodes * 0.40)  # after 40%
    phase3_start = int(episodes * 0.70)  # after 40% + 30%
    phase4_start = int(episodes * 0.90)  # after 40% + 30% + 20%

    path = _model_path(agent_id)

    agent = DQNAgent(agent_id)
    win_count = 0
    recent_rew = []

    if verbose:
        print(f"\n{'='*64}", flush=True)
        print(f"  DQN Training  |  Agent {agent_id}  |  {episodes} episodes", flush=True)
        print(f"  Phase 1  ep    1 – {phase2_start:4d}  vs Random      (40%)", flush=True)
        print(f"  Phase 2  ep {phase2_start+1:4d} – {phase3_start:4d}  vs Greedy-5    (30%)", flush=True)
        print(f"  Phase 3  ep {phase3_start+1:4d} – {phase4_start:4d}  vs Greedy-10   (20%)", flush=True)
        print(f"  Phase 4  ep {phase4_start+1:4d} – {episodes:4d}  vs Greedy-15   (10%)", flush=True)
        print(f"{'='*64}", flush=True)

    for ep in range(episodes):
        world = World()
        total_rew = 0.0
        last_transition = None

        # Decide opponent difficulty for this episode
        if ep < phase2_start:
            greedy_candidates = 0
            phase_label = "Random"
        elif ep < phase3_start:
            greedy_candidates = 5
            phase_label = "Greedy-5"
        elif ep < phase4_start:
            greedy_candidates = 10
            phase_label = "Greedy-10"
        else:
            greedy_candidates = 15
            phase_label = "Greedy-15"

        for _ in range(MAX_TURNS * 2):
            over, _ = world.is_game_over()
            if over:
                break

            if world.current_agent == agent_id:
                state_vec = agent.get_state_vector(world)
                pre_stability = world.stability
                moves = get_all_valid_moves(world, agent_id)

                if not moves:
                    simulate(world)
                    world.switch_agent()
                    continue

                action, r, c = agent.choose_action(world)
                action_idx = DQN_ACTION_INDEX.get(action.name, 0)

                apply_action(world, agent_id, action, r, c)
                simulate(world)
                world.switch_agent()

                over2, _ = world.is_game_over()

                raw_reward = compute_reward(pre_stability, world, agent_id)
                reward = max(-15.0, min(15.0, raw_reward))

                next_sv = agent.get_state_vector(world)
                agent.update(state_vec, action_idx, reward, next_sv, done=over2)
                total_rew += reward
                last_transition = (state_vec, action_idx, next_sv)

            else:
                _run_opponent_turn(world, opponent_id, greedy_candidates)

        # Terminal reward
        if last_transition is not None:
            winner = world.get_winner()
            if winner == agent_id:
                terminal_reward = 25.0
                win_count += 1
            elif winner is None:
                terminal_reward = 0.0
            else:
                terminal_reward = -25.0

            sv, aidx, nsv = last_transition
            agent.update(sv, aidx, terminal_reward, nsv, done=True)
        elif world.get_winner() == agent_id:
            win_count += 1

        recent_rew.append(total_rew)
        if len(recent_rew) > 100:
            recent_rew.pop(0)

        # Four-stage epsilon decay aligned with 4-phase curriculum:
        # Phase 1: 1.00 -> 0.25
        # Phase 2: 0.25 -> 0.15
        # Phase 3: 0.15 -> 0.08
        # Phase 4: 0.08 -> epsilon_min
        if ep < phase2_start:
            frac = (ep + 1) / max(phase2_start, 1)
            agent.epsilon = max(0.25, 1.0 - frac * (1.0 - 0.25))
        elif ep < phase3_start:
            frac = (ep - phase2_start + 1) / max(phase3_start - phase2_start, 1)
            agent.epsilon = max(0.15, 0.25 - frac * (0.25 - 0.15))
        elif ep < phase4_start:
            frac = (ep - phase3_start + 1) / max(phase4_start - phase3_start, 1)
            agent.epsilon = max(0.08, 0.15 - frac * (0.15 - 0.08))
        else:
            frac = (ep - phase4_start + 1) / max(episodes - phase4_start, 1)
            agent.epsilon = max(agent.epsilon_min,
                                0.08 - frac * (0.08 - agent.epsilon_min))

        if verbose and (ep + 1) % 100 == 0:
            avg_rew = sum(recent_rew) / len(recent_rew)
            win_rate = win_count / (ep + 1) * 100.0
            print(
                f"  ep {ep+1:4d}/{episodes}"
                f"  avg_r={avg_rew:+7.1f}"
                f"  win={win_rate:5.1f}%"
                f"  e={agent.epsilon:.3f}"
                f"  buf={len(agent.buffer):5d}"
                f"  [{phase_label}]",
                flush=True
            )

        # Checkpoint every 500 episodes
        if (ep + 1) % 500 == 0:
            agent.trained = True
            agent.save(path)
            if verbose:
                win_rate = win_count / (ep + 1) * 100.0
                print(
                    f"\n  [Checkpoint] ep {ep+1}  win={win_rate:.1f}%"
                    f"  e={agent.epsilon:.3f}  -> saved to {path}\n",
                    flush=True
                )

    # Final save
    agent.trained = True
    agent.save(path)

    if verbose:
        print(f"\n  Final win rate : {win_count/episodes*100:.1f}%  over {episodes} ep", flush=True)
        print(f"  Buffer size    : {len(agent.buffer)} transitions", flush=True)
        print(f"  Network updates: {agent.steps}", flush=True)
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
    parser.add_argument(
        '--agent',
        default='A',
        choices=['A', 'B'],
        help="Agent slot to train (default: A)"
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help="Training episodes (default: 5000)"
    )
    args = parser.parse_args()

    agent_id = AGENT_A if args.agent == 'A' else AGENT_B
    run_training(agent_id=agent_id, episodes=args.episodes)