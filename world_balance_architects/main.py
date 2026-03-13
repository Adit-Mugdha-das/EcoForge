# =============================================================================
# main.py — Entry point and game loop for World Balance Architects
# =============================================================================

import pygame
import sys

from config import *
from engine.world import World
from engine.simulate import simulate
from render.renderer import Renderer


def main():
    # ---- Pygame-CE initialization ----
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    # ---- Game state ----
    world    = World()
    renderer = Renderer(screen)
    game_over = False
    end_reason = None

    # Print ASCII grid to terminal so we can verify the layout immediately
    world.print_grid()
    print("\nControls: SPACE = next turn | R = reset | ESC = quit\n")

    # ---- Main loop ----
    running = True
    while running:

        # --- Event handling ---
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                # ESCAPE — quit
                if event.key == pygame.K_ESCAPE:
                    running = False

                # R — reset game
                if event.key == pygame.K_r:
                    world      = World()
                    game_over  = False
                    end_reason = None
                    world.print_grid()
                    print("Game reset.\n")

                # SPACE — advance one agent turn (placeholder until AI is ready)
                if event.key == pygame.K_SPACE and not game_over:
                    game_over, end_reason = _placeholder_turn(world)

        # --- Draw ---
        renderer.draw(world)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def _placeholder_turn(world: World):
    """
    Advance one agent's turn:
    1. Log a 'pass' action (placeholder for AI decision).
    2. Run the world simulation (water, crops, forests, meters).
    3. Switch active agent.
    4. Print terminal summary.
    5. Check game-over conditions.

    Returns (game_over: bool, reason: str | None).
    """
    prev_agent = world.current_agent

    # Placeholder: agent passes (no AI yet)
    world.log_action(f"Agent {prev_agent} passed")

    # Run simulation after each agent's turn
    simulate(world)

    # Switch active agent
    world.switch_agent()

    # Terminal summary every full round (after Agent B acts)
    if world.current_agent == AGENT_A:
        world.print_grid()

    # Check game-over
    over, reason = world.is_game_over()
    if over:
        winner = world.get_winner()
        print(f"\n{'='*40}")
        print(f"GAME OVER — {reason.upper()}")
        print(f"Winner: Agent {winner}" if winner else "Result: Draw")
        print(f"Final scores — A: {world.scores[AGENT_A]:.1f}  "
              f"B: {world.scores[AGENT_B]:.1f}")
        print(f"{'='*40}\n")
        return True, reason

    return False, None


if __name__ == "__main__":
    main()
