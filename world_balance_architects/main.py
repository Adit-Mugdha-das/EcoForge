# =============================================================================
# main.py — Entry point and game loop for World Balance Architects
# =============================================================================

import pygame
import sys

from config import *
from engine.world import World
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

    # Print ASCII grid to terminal so we can verify the layout immediately
    world.print_grid()

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

                # SPACE — manually advance one agent turn (placeholder)
                # This will later be replaced by AI agent calls
                if event.key == pygame.K_SPACE:
                    _placeholder_turn(world)

        # --- Draw ---
        renderer.draw(world)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


def _placeholder_turn(world: World):
    """
    Temporary function: just switches the active agent and logs it.
    This will be replaced by actual AI agent decision calls
    once the agents are implemented.
    """
    prev_agent = world.current_agent
    world.switch_agent()
    world.log_action(f"Agent {prev_agent} passed (no AI yet)")

    # Check for game over
    over, reason = world.is_game_over()
    if over:
        winner = world.get_winner()
        print(f"\nGame Over — {reason}!")
        if winner:
            print(f"Winner: Agent {winner}")
        else:
            print("Result: Draw")


if __name__ == "__main__":
    main()
