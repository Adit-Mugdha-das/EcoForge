# =============================================================================
# main.py — Entry point and game loop for World Balance Architects
# =============================================================================
#
# An in-game selection screen lets you pick Agent A and Agent B at startup.
# Available agents: minimax | montecarlo | qlearning | dqn
#
# Controls (in-game):
#   SPACE      — step one agent turn manually
#   A          — toggle auto-play (agents play continuously)
#   R          — reset / return to agent selection
#   ESC        — quit
#   +  /  -    — auto-play speed up / slow down
# =============================================================================

import pygame
import sys
import os

from config import *
from engine.world import World
from engine.simulate import simulate
from render.renderer import Renderer
from agents.minimax import MinimaxAgent
from agents.monte_carlo import MonteCarloAgent
from agents.q_learning import QLearningAgent, ACTION_INDEX
from agents.dqn_agent import DQNAgent, DQN_ACTION_INDEX
from agents.eval import compute_reward


# ── Auto-play delay in milliseconds between agent turns ──────────────────────
AUTO_PLAY_DELAY_MS  = 600     # default: one turn every 600 ms
AUTO_PLAY_SPEED_STEP = 100    # +/- key changes delay by this amount
AUTO_PLAY_MIN_MS     = 100
AUTO_PLAY_MAX_MS     = 2000


# =============================================================================
# Agent-selection screen (shown at startup and after every reset)
# =============================================================================

_AGENT_OPTIONS = [
    ('minimax',    'Minimax',     'Adversarial tree search'),
    ('montecarlo', 'Monte Carlo', 'Simulation-based rollouts'),
    ('qlearning',  'Q-Learning',  'Experience & learning'),
    ('dqn',        'DQN',         'Deep neural network RL'),
]


def run_selection_screen(screen, clock):
    """
    Display an interactive agent-selection screen.
    The user clicks one button per column to pick Agent A and Agent B,
    then clicks Start.  Returns (a_type, b_type) as strings.
    """
    # ── Fonts ──────────────────────────────────────────────────────────────────
    f_title  = pygame.font.SysFont('Segoe UI', 34, bold=True)
    f_sub    = pygame.font.SysFont('Segoe UI', 17)
    f_head   = pygame.font.SysFont('Segoe UI', 21, bold=True)
    f_btn    = pygame.font.SysFont('Segoe UI', 17, bold=True)
    f_desc   = pygame.font.SysFont('Segoe UI', 13)
    f_start  = pygame.font.SysFont('Segoe UI', 19, bold=True)

    # ── Layout ─────────────────────────────────────────────────────────────────
    COL_A_CX  = SCREEN_WIDTH // 4        # 225 — Agent A column centre
    COL_B_CX  = SCREEN_WIDTH * 3 // 4   # 675 — Agent B column centre
    BTN_W, BTN_H = 190, 58
    BTN_GAP      = 10
    FIRST_BTN_Y  = 175

    # ── Colours ────────────────────────────────────────────────────────────────
    BG          = (15, 18, 28)
    C_TEXT      = (220, 220, 220)
    C_DIM       = (130, 140, 160)
    C_DIV       = (45, 55, 75)
    C_BTN       = (35, 44, 60)
    C_BTN_HOV   = (50, 62, 85)
    C_SEL_A     = (130, 35, 35)
    C_SEL_B     = (28, 65, 140)
    C_BORDER    = (60, 70, 90)
    C_BORDER_SA = (220, 80, 80)
    C_BORDER_SB = (80, 130, 220)
    C_START     = (45, 140, 65)
    C_START_H   = (60, 170, 85)
    C_HEAD_A    = (220, 80, 80)
    C_HEAD_B    = (80, 130, 220)

    selected = {AGENT_A: 'minimax', AGENT_B: 'montecarlo'}

    while True:
        mx, my = pygame.mouse.get_pos()
        screen.fill(BG)

        # ── Title ──────────────────────────────────────────────────────────────
        t1 = f_title.render("World Balance Architects", True, C_TEXT)
        screen.blit(t1, (SCREEN_WIDTH // 2 - t1.get_width() // 2, 38))
        t2 = f_sub.render("Choose which agents will compete", True, C_DIM)
        screen.blit(t2, (SCREEN_WIDTH // 2 - t2.get_width() // 2, 84))

        # ── Divider + column headers ────────────────────────────────────────────
        pygame.draw.line(screen, C_DIV,
                         (SCREEN_WIDTH // 2, 115), (SCREEN_WIDTH // 2, 455), 1)

        ha = f_head.render("Agent A", True, C_HEAD_A)
        screen.blit(ha, (COL_A_CX - ha.get_width() // 2, 120))
        hb = f_head.render("Agent B", True, C_HEAD_B)
        screen.blit(hb, (COL_B_CX - hb.get_width() // 2, 120))

        # Thin underline beneath headers
        for cx, color in ((COL_A_CX, C_HEAD_A), (COL_B_CX, C_HEAD_B)):
            pygame.draw.line(screen, color,
                             (cx - 55, 145), (cx + 55, 145), 2)

        # ── Agent buttons (3 per column) ────────────────────────────────────────
        btn_map = {AGENT_A: [], AGENT_B: []}

        for agent_id, col_cx, c_sel, c_bord_sel in (
            (AGENT_A, COL_A_CX, C_SEL_A, C_BORDER_SA),
            (AGENT_B, COL_B_CX, C_SEL_B, C_BORDER_SB),
        ):
            for row, (atype, label, desc) in enumerate(_AGENT_OPTIONS):
                bx = col_cx - BTN_W // 2
                by = FIRST_BTN_Y + row * (BTN_H + BTN_GAP)
                rect = pygame.Rect(bx, by, BTN_W, BTN_H)
                btn_map[agent_id].append((rect, atype))

                is_sel = selected[agent_id] == atype
                is_hov = rect.collidepoint(mx, my) and not is_sel

                # Background
                fill   = c_sel if is_sel else (C_BTN_HOV if is_hov else C_BTN)
                border = c_bord_sel if is_sel else C_BORDER
                bw     = 2 if is_sel else 1
                pygame.draw.rect(screen, fill, rect, border_radius=8)
                pygame.draw.rect(screen, border, rect, bw, border_radius=8)

                # Label + description
                lt = f_btn.render(label, True, C_TEXT)
                screen.blit(lt, (rect.centerx - lt.get_width() // 2,
                                 rect.top + 14))
                dt = f_desc.render(desc, True,
                                   C_TEXT if is_sel else C_DIM)
                screen.blit(dt, (rect.centerx - dt.get_width() // 2,
                                 rect.top + 38))

        # ── Start button ────────────────────────────────────────────────────────
        start_rect = pygame.Rect(SCREEN_WIDTH // 2 - 130, 465, 260, 52)
        s_hov      = start_rect.collidepoint(mx, my)
        pygame.draw.rect(screen, C_START_H if s_hov else C_START,
                         start_rect, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, start_rect, 1, border_radius=10)
        st = f_start.render("Start Game", True, (255, 255, 255))
        screen.blit(st, (start_rect.centerx - st.get_width() // 2,
                         start_rect.centery - st.get_height() // 2))

        # ── Selection summary (below headers, above buttons) ────────────────────
        # (drawn implicitly via button highlight — no extra text needed)

        pygame.display.flip()
        clock.tick(FPS)

        # ── Events ─────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check agent buttons
                for agent_id, buttons in btn_map.items():
                    for rect, atype in buttons:
                        if rect.collidepoint(event.pos):
                            selected[agent_id] = atype
                # Check start
                if start_rect.collidepoint(event.pos):
                    return selected[AGENT_A], selected[AGENT_B]


def _q_table_path(agent_id: str) -> str:
    """Return the filename used to persist a Q-agent's table between runs."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"q_table_{agent_id}.json")


def _dqn_model_path(agent_id: str) -> str:
    """Return the regular (latest) model path."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"dqn_model_{agent_id}.pt")


def _dqn_best_model_path(agent_id: str) -> str:
    """Return the best-win-rate model path (saved separately during training)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"dqn_model_{agent_id}_best.pt")


def save_learners(agents: dict):
    """Save Q-tables and DQN models so all online experience is kept."""
    for agent_id, agent in agents.items():
        if isinstance(agent, QLearningAgent):
            agent.save(_q_table_path(agent_id))
        elif isinstance(agent, DQNAgent) and agent.trained:
            agent.save(_dqn_model_path(AGENT_A))  # always write to A's file


def build_agent(agent_type: str, agent_id: str):
    """Instantiate and (if needed) train an agent of the given type."""
    agent_type = agent_type.lower().strip()

    if agent_type == 'minimax':
        return MinimaxAgent(agent_id, depth=2, max_branches=12)

    elif agent_type == 'montecarlo':
        return MonteCarloAgent(agent_id, num_rollouts=15, rollout_depth=8)

    elif agent_type == 'qlearning':
        agent = QLearningAgent(agent_id)
        path  = _q_table_path(agent_id)
        if os.path.exists(path):
            agent.load(path)
        else:
            agent.train(episodes=TRAIN_EPISODES, verbose=True)
            agent.save(path)
        return agent

    elif agent_type == 'dqn':
        agent = DQNAgent(agent_id)

        # Always load from Agent A's model — it is the only trained model.
        # Both Agent A and Agent B slots share the same weights.
        # The state vector is fully relative (own_eco, opp_eco, own_cells,
        # opp_cells, score_diff) so the same weights work correctly from
        # either agent's perspective.
        best_path = _dqn_best_model_path(AGENT_A)   # dqn_model_A_best.pt
        path      = _dqn_model_path(AGENT_A)         # dqn_model_A.pt
        load_path = best_path if os.path.exists(best_path) else path

        if os.path.exists(load_path):
            which = "best" if "_best" in load_path else "latest"
            print(f"[DQN] Agent {agent_id} loading {which} model: {load_path}")
            try:
                agent.load(load_path)
            except RuntimeError as exc:
                print(f"[DQN] WARNING: {exc}")
                print(f"[DQN] Starting untrained — run: python train_dqn.py")
        else:
            print(f"[DQN] No trained model found (expected {path}).")
            print(f"[DQN] Run:  python train_dqn.py")
            print(f"[DQN] Starting untrained — agent will explore randomly.")
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

    # Capture pre-action state for online learning (Q-Learning and DQN)
    is_qlearner   = isinstance(agent, QLearningAgent)
    is_dqn        = isinstance(agent, DQNAgent)
    is_learner    = is_qlearner or is_dqn

    if is_qlearner:
        pre_state = world.get_state_category()
    elif is_dqn:
        pre_state = agent.get_state_vector(world)
    else:
        pre_state = None

    pre_stability = world.stability if is_learner else None
    action_idx    = None

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
        if is_qlearner:
            action_idx = ACTION_INDEX.get(action.name, 0)
        elif is_dqn:
            action_idx = DQN_ACTION_INDEX.get(action.name, 0)
        try:
            from engine.actions import apply_action
            log = apply_action(world, current, action, r, c)
            print(f"  {log}")
        except Exception as exc:
            print(f"  [WARN] Agent {current} apply_action raised: {exc} — skipping turn")
            world.log_action(f"Agent {current} ({agent.name}): action failed")
            action_idx = None   # action failed — don't update Q-table

    # Simulate the world after this agent's action
    try:
        simulate(world)
    except Exception as exc:
        print(f"  [WARN] simulate() raised: {exc} — world state may be inconsistent")

    # Online learning update (runs after simulate so next_state reflects changes)
    if is_learner and pre_state is not None and action_idx is not None:
        reward = compute_reward(pre_stability, world, current)
        if is_qlearner:
            next_state = world.get_state_category()
            agent.update(pre_state, action_idx, reward, next_state)
        else:  # DQN
            next_state = agent.get_state_vector(world)
            agent.update(pre_state, action_idx, reward, next_state)

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
    # ── Pygame-CE init (must come before the selection screen) ────────────────
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    # Outer loop: selection → game → back to selection on R
    while True:
        # ── Agent selection screen ─────────────────────────────────────────────
        a_type, b_type = run_selection_screen(screen, clock)

        # ── Build agents ───────────────────────────────────────────────────────
        print(f"\nBuilding agents:  A={a_type}  B={b_type}")
        agent_a = build_agent(a_type, AGENT_A)
        agent_b = build_agent(b_type, AGENT_B)
        agents  = {AGENT_A: agent_a, AGENT_B: agent_b}
        print(f"  Agent A → {agent_a.name}")
        print(f"  Agent B → {agent_b.name}\n")

        pygame.display.set_caption(
            f"{TITLE}  |  A: {agent_a.name}  vs  B: {agent_b.name}"
        )

        # ── Game state ─────────────────────────────────────────────────────────
        world         = World()
        renderer      = Renderer(screen)
        game_over     = False
        end_reason    = None
        auto_play     = True
        auto_delay_ms = AUTO_PLAY_DELAY_MS
        last_auto_ms  = 0

        world.print_grid()
        print("Game started — agents are playing automatically.")
        print("Controls: SPACE=step  A=pause/resume  R=new selection  +/-=speed  ESC=quit\n")

        # ── Game loop ──────────────────────────────────────────────────────────
        running = True
        while running:
            now_ms = pygame.time.get_ticks()

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    save_learners(agents)
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        save_learners(agents)
                        pygame.quit()
                        sys.exit()

                    # Toggle auto-play pause/resume
                    elif event.key == pygame.K_a:
                        auto_play = not auto_play
                        print(f"Auto-play: {'RESUMED' if auto_play else 'PAUSED'}")

                    # Manual step
                    elif event.key == pygame.K_SPACE and not game_over:
                        game_over, end_reason = run_agent_turn(world, agents)

                    # R — save, then return to agent-selection screen
                    elif event.key == pygame.K_r:
                        save_learners(agents)
                        running = False

                    # Speed up / slow down auto-play
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        auto_delay_ms = max(AUTO_PLAY_MIN_MS,
                                            auto_delay_ms - AUTO_PLAY_SPEED_STEP)
                        print(f"Auto-play delay: {auto_delay_ms} ms")

                    elif event.key == pygame.K_MINUS:
                        auto_delay_ms = min(AUTO_PLAY_MAX_MS,
                                            auto_delay_ms + AUTO_PLAY_SPEED_STEP)
                        print(f"Auto-play delay: {auto_delay_ms} ms")

            # ── Auto-play tick ────────────────────────────────────────────────
            if auto_play and not game_over:
                if now_ms - last_auto_ms >= auto_delay_ms:
                    game_over, end_reason = run_agent_turn(world, agents)
                    last_auto_ms = now_ms
                    if game_over:
                        auto_play = False

            # ── Draw ──────────────────────────────────────────────────────────
            renderer.draw(world)
            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
