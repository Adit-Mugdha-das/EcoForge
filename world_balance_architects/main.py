# =============================================================================
# main.py — Entry point and game loop for World Balance Architects
# =============================================================================
#
# AGENT SELECTION — change these two lines to pick which agents compete:
#   'minimax'      → MinimaxAgent  (adversarial planning)
#   'montecarlo'   → MonteCarloAgent (simulation-based)
#   'qlearning'    → QLearningAgent (experience-based, trains before game)
#
AGENT_A_TYPE = 'minimax'
AGENT_B_TYPE = 'montecarlo'
#
# Controls:
#   SPACE      — step one agent turn manually
#   A          — toggle auto-play (agents play continuously)
#   R          — reset / start new game
#   ESC        — quit
#   +  /  -    — auto-play speed up / slow down
# =============================================================================

import pygame
import sys

from config import *
from engine.world import World
from engine.simulate import simulate
from render.renderer import Renderer
from agents.minimax import MinimaxAgent
from agents.monte_carlo import MonteCarloAgent
from agents.q_learning import QLearningAgent


# ── Auto-play delay in milliseconds between agent turns ──────────────────────
AUTO_PLAY_DELAY_MS  = 600     # default: one turn every 600 ms
AUTO_PLAY_SPEED_STEP = 100    # +/- key changes delay by this amount
AUTO_PLAY_MIN_MS     = 100
AUTO_PLAY_MAX_MS     = 2000


def build_agent(agent_type: str, agent_id: str):
    """Instantiate and (if needed) train an agent of the given type."""
    agent_type = agent_type.lower().strip()

    if agent_type == 'minimax':
        return MinimaxAgent(agent_id, depth=2, max_branches=12)

    elif agent_type == 'montecarlo':
        return MonteCarloAgent(agent_id, num_rollouts=15, rollout_depth=8)

    elif agent_type == 'qlearning':
        agent = QLearningAgent(agent_id)
        agent.train(episodes=TRAIN_EPISODES, verbose=True)
        return agent

    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'. "
                         f"Choose from: minimax, montecarlo, qlearning")


def run_agent_turn(world: World, agents: dict) -> tuple:
    """
    Let the current active agent choose and apply its action,
    run world simulation, and switch turns.

    Returns (game_over: bool, reason: str | None).
    """
    current    = world.current_agent
    agent      = agents[current]

    # Agent chooses its best action — guarded so a crash doesn't kill pygame
    try:
        action, r, c = agent.choose_action(world)
    except Exception as exc:
        print(f"  [WARN] Agent {current} choose_action raised: {exc} — skipping turn")
        action, r, c = None, -1, -1

    if action is None:
        # No valid moves (or error above) — agent passes this turn
        world.log_action(f"Agent {current} ({agent.name}): no valid moves")
    else:
        try:
            from engine.actions import apply_action
            log = apply_action(world, current, action, r, c)
            print(f"  {log}")
        except Exception as exc:
            print(f"  [WARN] Agent {current} apply_action raised: {exc} — skipping turn")
            world.log_action(f"Agent {current} ({agent.name}): action failed")

    # Simulate the world after this agent's action
    try:
        simulate(world)
    except Exception as exc:
        print(f"  [WARN] simulate() raised: {exc} — world state may be inconsistent")

    # Switch active agent
    world.switch_agent()

    # Print summary every full round
    if world.current_agent == AGENT_A:
        world.print_grid()

    # Check game-over
    over, reason = world.is_game_over()
    if over:
        winner = world.get_winner()
        print(f"\n{'='*42}")
        print(f" GAME OVER — {reason.upper()}")
        if winner:
            print(f" Winner: Agent {winner}  ({agents[winner].name})")
        else:
            print(" Result: Draw")
        print(f" Scores — A: {world.scores[AGENT_A]:.1f}  "
              f"B: {world.scores[AGENT_B]:.1f}")
        print(f"{'='*42}\n")
        return True, reason

    return False, None


def main():
    # ── Build agents ──────────────────────────────────────────────────────────
    print(f"\nBuilding agents:  A={AGENT_A_TYPE}  B={AGENT_B_TYPE}")
    agent_a = build_agent(AGENT_A_TYPE, AGENT_A)
    agent_b = build_agent(AGENT_B_TYPE, AGENT_B)
    agents  = {AGENT_A: agent_a, AGENT_B: agent_b}
    print(f"  Agent A → {agent_a.name}")
    print(f"  Agent B → {agent_b.name}\n")

    # ── Pygame-CE init ────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(
        f"{TITLE}  |  A: {agent_a.name}  vs  B: {agent_b.name}"
    )
    clock = pygame.time.Clock()

    # ── Game state ────────────────────────────────────────────────────────────
    world         = World()
    renderer      = Renderer(screen)
    game_over     = False
    end_reason    = None
    auto_play     = True          # starts playing immediately on launch
    auto_delay_ms = AUTO_PLAY_DELAY_MS
    last_auto_ms  = 0

    world.print_grid()
    print("Game started — agents are playing automatically.")
    print("Controls: SPACE=step  A=pause/resume  R=reset  +/-=speed  ESC=quit\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        now_ms = pygame.time.get_ticks()

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                # Toggle auto-play pause/resume
                elif event.key == pygame.K_a:
                    auto_play = not auto_play
                    print(f"Auto-play: {'RESUMED' if auto_play else 'PAUSED'}")

                # Manual step
                elif event.key == pygame.K_SPACE and not game_over:
                    game_over, end_reason = run_agent_turn(world, agents)

                # Reset — starts a brand new game, auto-play resumes
                elif event.key == pygame.K_r:
                    world      = World()
                    game_over  = False
                    end_reason = None
                    auto_play  = True
                    world.print_grid()
                    print("Game reset — agents playing again.\n")

                # Speed up / slow down auto-play
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    auto_delay_ms = max(AUTO_PLAY_MIN_MS,
                                        auto_delay_ms - AUTO_PLAY_SPEED_STEP)
                    print(f"Auto-play delay: {auto_delay_ms} ms")

                elif event.key == pygame.K_MINUS:
                    auto_delay_ms = min(AUTO_PLAY_MAX_MS,
                                        auto_delay_ms + AUTO_PLAY_SPEED_STEP)
                    print(f"Auto-play delay: {auto_delay_ms} ms")

        # ── Auto-play tick ────────────────────────────────────────────────────
        if auto_play and not game_over:
            if now_ms - last_auto_ms >= auto_delay_ms:
                game_over, end_reason = run_agent_turn(world, agents)
                last_auto_ms = now_ms
                if game_over:
                    auto_play = False

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw(world)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
