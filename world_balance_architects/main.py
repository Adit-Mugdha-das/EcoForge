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
import threading

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


def _load_agent_images(card_image_size: int = 92) -> dict[str, pygame.Surface | None]:
    """Load character images for each agent option from assets/agent_images."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "assets", "agent_images")
    loaded: dict[str, pygame.Surface | None] = {k: None for k, _l, _d in _AGENT_OPTIONS}

    if not os.path.isdir(images_dir):
        return loaded

    files = [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    for atype, _label, _desc in _AGENT_OPTIONS:
        keyword = atype.replace("_", "")
        match = None
        for fname in files:
            normalized = fname.lower().replace("_", "").replace(" ", "")
            if keyword in normalized:
                match = fname
                break
        if match is None and atype == "montecarlo":
            for fname in files:
                normalized = fname.lower().replace("_", "").replace(" ", "")
                if "montecarlo" in normalized or "monte" in normalized:
                    match = fname
                    break

        if match is not None:
            path = os.path.join(images_dir, match)
            try:
                surf = pygame.image.load(path).convert_alpha()
                surf = pygame.transform.smoothscale(surf, (card_image_size, card_image_size))
                loaded[atype] = surf
            except Exception:
                loaded[atype] = None

    return loaded


def run_selection_screen(screen, clock):
    """
    Display a two-page interactive agent-selection screen.
    Page 1 selects Agent A. Page 2 selects Agent B.
    Returns (a_type, b_type) as strings.
    """
    f_title   = pygame.font.SysFont('Segoe UI', 34, bold=True)
    f_sub     = pygame.font.SysFont('Segoe UI', 17)
    f_step    = pygame.font.SysFont('Segoe UI', 16, bold=True)
    f_btn     = pygame.font.SysFont('Segoe UI', 18, bold=True)
    f_desc    = pygame.font.SysFont('Segoe UI', 13)
    f_small   = pygame.font.SysFont('Segoe UI', 12)

    BG            = (13, 16, 26)
    C_TEXT        = (224, 232, 245)
    C_DIM         = (132, 145, 168)
    C_BORDER      = (66, 80, 105)
    C_CARD        = (28, 37, 55)
    C_CARD_HOV    = (40, 53, 78)
    C_CARD_SEL_A  = (95, 42, 48)
    C_CARD_SEL_B  = (40, 62, 100)
    C_ACCENT_A    = (235, 96, 96)
    C_ACCENT_B    = (102, 154, 240)
    C_NEXT        = (46, 138, 84)
    C_NEXT_H      = (64, 166, 101)
    C_BACK        = (63, 74, 100)
    C_BACK_H      = (82, 94, 123)

    CARD_W, CARD_H = 430, 132
    GRID_X = (SCREEN_WIDTH - (CARD_W * 2 + 26)) // 2
    GRID_Y = 170

    images = _load_agent_images(card_image_size=92)
    selected = {AGENT_A: 'minimax', AGENT_B: 'montecarlo'}

    page = 0
    pending_page = 0
    fade_alpha = 0
    fade_phase = None  # None | 'out' | 'in'
    anim_t = 0.0

    while True:
        dt = clock.tick(FPS)
        anim_t += dt * 0.001
        mx, my = pygame.mouse.get_pos()

        # Soft animated background glow
        screen.fill(BG)
        glow = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        g1 = int(42 + 14 * (0.5 + 0.5 * pygame.math.Vector2(1, 0).rotate(anim_t * 35).x))
        g2 = int(38 + 18 * (0.5 + 0.5 * pygame.math.Vector2(1, 0).rotate(anim_t * 28 + 30).x))
        pygame.draw.circle(glow, (40, 70, 120, g1), (180, 120), 180)
        pygame.draw.circle(glow, (160, 70, 70, g2), (SCREEN_WIDTH - 180, 140), 180)
        screen.blit(glow, (0, 0))

        role = AGENT_A if page == 0 else AGENT_B
        role_label = 'Agent A' if page == 0 else 'Agent B'
        accent = C_ACCENT_A if page == 0 else C_ACCENT_B
        card_sel = C_CARD_SEL_A if page == 0 else C_CARD_SEL_B

        # Header
        t1 = f_title.render('World Balance Architects', True, C_TEXT)
        screen.blit(t1, (SCREEN_WIDTH // 2 - t1.get_width() // 2, 28))
        subtitle = f'Step {page + 1}/2  -  Choose {role_label}'
        t2 = f_sub.render(subtitle, True, C_DIM)
        screen.blit(t2, (SCREEN_WIDTH // 2 - t2.get_width() // 2, 75))

        # Progress pill
        pill = pygame.Rect(SCREEN_WIDTH // 2 - 90, 104, 180, 28)
        pygame.draw.rect(screen, (24, 31, 45), pill, border_radius=14)
        pygame.draw.rect(screen, C_BORDER, pill, 1, border_radius=14)
        prog_w = int((pill.width - 6) * ((page + 1) / 2.0))
        prog = pygame.Rect(pill.x + 3, pill.y + 3, prog_w, pill.height - 6)
        pygame.draw.rect(screen, accent, prog, border_radius=12)
        step_txt = f_step.render(role_label, True, C_TEXT)
        screen.blit(step_txt, (pill.centerx - step_txt.get_width() // 2,
                               pill.centery - step_txt.get_height() // 2))

        # Cards (2 x 2)
        card_buttons = []
        for idx, (atype, label, desc) in enumerate(_AGENT_OPTIONS):
            col = idx % 2
            row = idx // 2
            x = GRID_X + col * (CARD_W + 26)
            y = GRID_Y + row * (CARD_H + 18)

            bob = int(3.0 * pygame.math.Vector2(1, 0).rotate(anim_t * 100 + idx * 33).x)
            rect = pygame.Rect(x, y + bob, CARD_W, CARD_H)
            is_sel = selected[role] == atype
            is_hov = rect.collidepoint(mx, my)

            fill = card_sel if is_sel else (C_CARD_HOV if is_hov else C_CARD)
            border = accent if is_sel else C_BORDER
            border_w = 2 if is_sel else 1
            pygame.draw.rect(screen, fill, rect, border_radius=14)
            pygame.draw.rect(screen, border, rect, border_w, border_radius=14)

            img = images.get(atype)
            img_box = pygame.Rect(rect.x + 14, rect.y + 20, 92, 92)
            pygame.draw.rect(screen, (20, 26, 38), img_box, border_radius=10)
            if img is not None:
                screen.blit(img, (img_box.x, img_box.y))
            else:
                tag = f_small.render(label[:2].upper(), True, C_TEXT)
                screen.blit(tag, (img_box.centerx - tag.get_width() // 2,
                                  img_box.centery - tag.get_height() // 2))

            lt = f_btn.render(label, True, C_TEXT)
            dtxt = f_desc.render(desc, True, C_TEXT if is_sel else C_DIM)
            screen.blit(lt, (img_box.right + 18, rect.y + 34))
            screen.blit(dtxt, (img_box.right + 18, rect.y + 64))

            card_buttons.append((rect, atype))

        # Selection summary row
        summary = f"A: {selected[AGENT_A]}    vs    B: {selected[AGENT_B]}"
        sm = f_sub.render(summary, True, C_DIM)
        screen.blit(sm, (SCREEN_WIDTH // 2 - sm.get_width() // 2, 462))

        # Footer buttons
        back_rect = pygame.Rect(SCREEN_WIDTH // 2 - 210, 500, 170, 50)
        next_rect = pygame.Rect(SCREEN_WIDTH // 2 + 40, 500, 170, 50)

        if page == 1:
            back_hov = back_rect.collidepoint(mx, my)
            pygame.draw.rect(screen, C_BACK_H if back_hov else C_BACK, back_rect, border_radius=10)
            pygame.draw.rect(screen, C_BORDER, back_rect, 1, border_radius=10)
            bt = f_btn.render('Back', True, C_TEXT)
            screen.blit(bt, (back_rect.centerx - bt.get_width() // 2,
                             back_rect.centery - bt.get_height() // 2))

        next_hov = next_rect.collidepoint(mx, my)
        pygame.draw.rect(screen, C_NEXT_H if next_hov else C_NEXT, next_rect, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, next_rect, 1, border_radius=10)
        next_label = 'Next' if page == 0 else 'Start Match'
        nt = f_btn.render(next_label, True, (255, 255, 255))
        screen.blit(nt, (next_rect.centerx - nt.get_width() // 2,
                         next_rect.centery - nt.get_height() // 2))

        # Page transition fade
        if fade_phase == 'out':
            fade_alpha = min(255, fade_alpha + 26)
            if fade_alpha >= 255:
                page = pending_page
                fade_phase = 'in'
        elif fade_phase == 'in':
            fade_alpha = max(0, fade_alpha - 26)
            if fade_alpha <= 0:
                fade_phase = None

        if fade_alpha > 0:
            fade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            fade.fill((8, 10, 16, fade_alpha))
            screen.blit(fade, (0, 0))

        pygame.display.flip()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and fade_phase is None:
                for rect, atype in card_buttons:
                    if rect.collidepoint(event.pos):
                        selected[role] = atype

                if page == 1 and back_rect.collidepoint(event.pos):
                    pending_page = 0
                    fade_phase = 'out'

                if next_rect.collidepoint(event.pos):
                    if page == 0:
                        pending_page = 1
                        fade_phase = 'out'
                    else:
                        return selected[AGENT_A], selected[AGENT_B]


# =============================================================================
# Parameter-customisation screen (shown after agent selection)
# =============================================================================

# (key, label, default, vmin, vmax, step, hint)
_PARAMS_CONFIG = [
    ('water_level', 'Water Level',          25.0,  0.0, 100.0,  5.0, 'Optimal: 50 – 80'),
    ('food',        'Food',                 35.0,  0.0, 100.0,  5.0, 'Optimal: 50 – 80'),
    ('oxygen',      'Oxygen',               35.0,  0.0, 100.0,  5.0, 'Optimal: 50 – 80'),
    ('temperature', 'Temperature',          72.0,  0.0, 100.0,  5.0, 'Optimal: 40 – 60'),
    ('population',  'Population',           50.0,  0.0, 200.0, 10.0, 'Range: 0 – 200'),
    ('eco_points',  'Starting Eco (each)',  25.0,  5.0,  99.0,  5.0, 'Per agent'),
]

_MINIMAX_PARAMS = [
    ('mm_depth',    'Minimax Depth',         1.0,  1.0,   6.0,  1.0, 'Higher = stronger (slower)'),
    ('mm_branches', 'Max Branches',         12.0,  4.0,  28.0,  2.0, 'Moves sampled per node'),
]

_MC_PARAMS = [
    ('mc_rollouts', 'MC Rollouts',          10.0,  5.0,  60.0,  5.0, 'More = stronger (slower)'),
    ('mc_depth',    'MC Rollout Depth',      5.0,  4.0,  24.0,  2.0, 'Steps per rollout'),
]


def run_params_screen(screen, clock, a_type: str = '', b_type: str = ''):
    """
    Let the user tweak starting planet parameters and (if relevant)
    algorithm hyperparameters with +/- buttons.
    'Use Defaults' resets everything; 'Start Game' confirms.
    Returns a dict {key: value} used by World() and build_agent().
    """
    f_title  = pygame.font.SysFont('Segoe UI', 34, bold=True)
    f_sub    = pygame.font.SysFont('Segoe UI', 17)
    f_label  = pygame.font.SysFont('Segoe UI', 15, bold=True)
    f_sec    = pygame.font.SysFont('Segoe UI', 13, bold=True)
    f_val    = pygame.font.SysFont('Segoe UI', 15, bold=True)
    f_hint   = pygame.font.SysFont('Segoe UI', 12)
    f_btn    = pygame.font.SysFont('Segoe UI', 17, bold=True)
    f_start  = pygame.font.SysFont('Segoe UI', 19, bold=True)

    BG        = (15,  18,  28)
    C_TEXT    = (220, 220, 220)
    C_DIM     = (130, 140, 160)
    C_DIV     = (45,  55,  75)
    C_ROW     = (28,  35,  50)
    C_ROW_ALT = (22,  28,  42)
    C_ROW_ALG = (25,  40,  35)    # tinted green for algo rows
    C_ROW_ALG2= (20,  33,  28)
    C_BORDER  = (60,  70,  90)
    C_VAL     = (40,  50,  70)
    C_MINUS   = (110, 45,  45)
    C_MINUS_H = (145, 65,  65)
    C_PLUS    = (40,  105, 55)
    C_PLUS_H  = (55,  135, 75)
    C_DEF     = (55,  55,  80)
    C_DEF_H   = (80,  80,  120)
    C_START   = (45,  140, 65)
    C_START_H = (60,  170, 85)
    C_SEC     = (100, 160, 120)   # section separator label colour

    # Build combined list: planet rows always; algo rows if type selected
    algo_configs = []
    if 'minimax' in (a_type, b_type):
        algo_configs.extend(_MINIMAX_PARAMS)
    if 'montecarlo' in (a_type, b_type):
        algo_configs.extend(_MC_PARAMS)

    all_configs = list(_PARAMS_CONFIG) + algo_configs
    n_planet    = len(_PARAMS_CONFIG)
    has_algo    = len(algo_configs) > 0
    n_total     = len(all_configs)

    # Working values initialised to defaults
    values = {key: default for key, _l, default, _mn, _mx, _s, _h in all_configs}

    # ── Dynamic layout ─────────────────────────────────────────────────────────
    ROW_Y0  = 118
    SEP_H   = 26 if has_algo else 0     # section-separator height
    BTN_H   = 48
    BTN_GAP = 14                        # gap between last row and buttons
    # Available vertical space for rows
    _avail  = SCREEN_HEIGHT - ROW_Y0 - SEP_H - BTN_H - BTN_GAP - 10
    ROW_H   = max(40, min(62, _avail // n_total))
    BTN_SZ  = min(30, ROW_H - 10)

    # Horizontal positions (fixed regardless of row count)
    LABEL_X = 72
    MINUS_X = 490
    GAP     = 5
    VAL_W   = 72
    HINT_X  = 652

    def _row_y(i: int) -> int:
        """Top-left y of row i, accounting for the section separator."""
        if i < n_planet:
            return ROW_Y0 + i * ROW_H
        return ROW_Y0 + n_planet * ROW_H + SEP_H + (i - n_planet) * ROW_H

    def _algo_row_color(i: int):
        return C_ROW_ALG if (i - n_planet) % 2 == 0 else C_ROW_ALG2

    def _btn_area_top() -> int:
        return _row_y(n_total - 1) + ROW_H + BTN_GAP

    while True:
        mx, my = pygame.mouse.get_pos()
        screen.fill(BG)

        # Title
        t1 = f_title.render("Configure Starting Parameters", True, C_TEXT)
        screen.blit(t1, (SCREEN_WIDTH // 2 - t1.get_width() // 2, 30))
        t2 = f_sub.render(
            "Adjust the planet's initial state — or keep the defaults", True, C_DIM)
        screen.blit(t2, (SCREEN_WIDTH // 2 - t2.get_width() // 2, 75))
        pygame.draw.line(screen, C_DIV, (50, 108), (850, 108), 1)

        # Buttons (anchored below the last row)
        btn_y      = _btn_area_top()
        def_rect   = pygame.Rect(SCREEN_WIDTH // 2 - 250, btn_y, 190, BTN_H)
        start_rect = pygame.Rect(SCREEN_WIDTH // 2 +  60, btn_y, 190, BTN_H)

        pygame.draw.line(screen, C_DIV, (50, btn_y - 8), (850, btn_y - 8), 1)

        d_hov = def_rect.collidepoint(mx, my)
        pygame.draw.rect(screen, C_DEF_H if d_hov else C_DEF, def_rect, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, def_rect, 1, border_radius=10)
        dt = f_btn.render("Use Defaults", True, C_TEXT)
        screen.blit(dt, (def_rect.centerx  - dt.get_width()  // 2,
                         def_rect.centery  - dt.get_height() // 2))

        s_hov = start_rect.collidepoint(mx, my)
        pygame.draw.rect(screen, C_START_H if s_hov else C_START, start_rect, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, start_rect, 1, border_radius=10)
        st = f_start.render("Start Game", True, (255, 255, 255))
        screen.blit(st, (start_rect.centerx - st.get_width()  // 2,
                         start_rect.centery - st.get_height() // 2))

        # Section separator between planet and algo rows
        if has_algo:
            sep_y  = _row_y(n_planet)
            pygame.draw.line(screen, C_DIV, (50, sep_y - 4), (850, sep_y - 4), 1)
            sec_lbl = f_sec.render("Algorithm Settings", True, C_SEC)
            screen.blit(sec_lbl, (SCREEN_WIDTH // 2 - sec_lbl.get_width() // 2,
                                  sep_y - SEP_H + 4))

        # Parameter rows
        minus_btns = []
        plus_btns  = []

        for i, (key, label, default, vmin, vmax, step, hint) in enumerate(all_configs):
            row_y = _row_y(i)
            is_algo = i >= n_planet

            # Row background
            if is_algo:
                bg_color = _algo_row_color(i)
            else:
                bg_color = C_ROW if i % 2 == 0 else C_ROW_ALT
            pygame.draw.rect(screen, bg_color,
                             pygame.Rect(50, row_y, 800, ROW_H - 4), border_radius=6)

            # Label
            lt = f_label.render(label, True, C_TEXT)
            screen.blit(lt, (LABEL_X,
                             row_y + (ROW_H - 4) // 2 - lt.get_height() // 2))

            btn_y_row = row_y + ((ROW_H - 4) - BTN_SZ) // 2

            # [-]
            m_rect = pygame.Rect(MINUS_X, btn_y_row, BTN_SZ, BTN_SZ)
            m_hov  = m_rect.collidepoint(mx, my)
            pygame.draw.rect(screen, C_MINUS_H if m_hov else C_MINUS, m_rect, border_radius=5)
            pygame.draw.rect(screen, C_BORDER, m_rect, 1, border_radius=5)
            mt = f_val.render("−", True, C_TEXT)
            screen.blit(mt, (m_rect.centerx - mt.get_width() // 2,
                             m_rect.centery - mt.get_height() // 2))
            minus_btns.append((m_rect, key, vmin, step))

            # Value box
            v_rect = pygame.Rect(MINUS_X + BTN_SZ + GAP, btn_y_row, VAL_W, BTN_SZ)
            pygame.draw.rect(screen, C_VAL,    v_rect, border_radius=4)
            pygame.draw.rect(screen, C_BORDER, v_rect, 1, border_radius=4)
            v   = values[key]
            txt = str(int(v)) if v == int(v) else f"{v:.1f}"
            vt  = f_val.render(txt, True, C_TEXT)
            screen.blit(vt, (v_rect.centerx - vt.get_width() // 2,
                             v_rect.centery - vt.get_height() // 2))

            # [+]
            p_rect = pygame.Rect(v_rect.right + GAP, btn_y_row, BTN_SZ, BTN_SZ)
            p_hov  = p_rect.collidepoint(mx, my)
            pygame.draw.rect(screen, C_PLUS_H if p_hov else C_PLUS, p_rect, border_radius=5)
            pygame.draw.rect(screen, C_BORDER, p_rect, 1, border_radius=5)
            pt = f_val.render("+", True, C_TEXT)
            screen.blit(pt, (p_rect.centerx - pt.get_width() // 2,
                             p_rect.centery - pt.get_height() // 2))
            plus_btns.append((p_rect, key, vmax, step))

            # Hint
            ht = f_hint.render(hint, True, C_DIM)
            screen.blit(ht, (HINT_X,
                             row_y + (ROW_H - 4) // 2 - ht.get_height() // 2))

        pygame.display.flip()
        clock.tick(FPS)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for m_rect, key, vmin, step in minus_btns:
                    if m_rect.collidepoint(event.pos):
                        cfg = next(c for c in all_configs if c[0] == key)
                        values[key] = max(cfg[3], round(values[key] - step, 1))
                for p_rect, key, vmax, step in plus_btns:
                    if p_rect.collidepoint(event.pos):
                        cfg = next(c for c in all_configs if c[0] == key)
                        values[key] = min(cfg[4], round(values[key] + step, 1))
                if def_rect.collidepoint(event.pos):
                    values = {key: default
                              for key, _l, default, _mn, _mx, _s, _h in all_configs}
                if start_rect.collidepoint(event.pos):
                    return dict(values)


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


def build_agent(agent_type: str, agent_id: str, agent_params: dict = None):
    """Instantiate and (if needed) train an agent of the given type."""
    agent_type = agent_type.lower().strip()
    p = agent_params or {}

    if agent_type == 'minimax':
        return MinimaxAgent(agent_id,
                            depth=int(p.get('mm_depth', 2)),
                            max_branches=int(p.get('mm_branches', 12)))

    elif agent_type == 'montecarlo':
        return MonteCarloAgent(agent_id,
                               num_rollouts=int(p.get('mc_rollouts', 15)),
                               rollout_depth=int(p.get('mc_depth', 8)))

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


_ACTION_FX_KEYS = {
    'PlantForest': 'plant_forest',
    'BuildCanal': 'build_canal',
    'BuildReservoir': 'plant_reservoir',
    'PlantFarm': 'plant_crop',
    'HarvestCrop': 'harvest_crop',
    'ClearForest': 'clear_forest',
    'BuildSolarPlant': 'build_solar',
}


def _get_action_fx_key(action) -> str | None:
    return _ACTION_FX_KEYS.get(action.__class__.__name__)


class TurnExecutor:
    """Runs a game turn in a background thread and reports results."""
    def __init__(self):
        self.thread: threading.Thread | None = None
        self.result: tuple | None = None
        self.error: Exception | None = None
        self.is_running = False

    def start(self, world: World, agents: dict, renderer: Renderer | None = None):
        """Start a turn execution in background thread."""
        if self.is_running:
            return  # Already running
        self.result = None
        self.error = None
        self.is_running = True
        self.thread = threading.Thread(
            target=self._execute_turn,
            args=(world, agents, renderer),
            daemon=True
        )
        self.thread.start()

    def _execute_turn(self, world: World, agents: dict, renderer: Renderer | None):
        """Execute turn and store result."""
        try:
            self.result = run_agent_turn(world, agents, renderer)
        except Exception as e:
            self.error = e
        finally:
            self.is_running = False

    def is_done(self) -> bool:
        """Check if turn execution finished."""
        return not self.is_running

    def get_result(self) -> tuple:
        """Get the turn result. Only call after is_done() returns True."""
        if self.error:
            raise self.error
        return self.result or (False, None)


def run_agent_turn(world: World, agents: dict, renderer: Renderer | None = None) -> tuple:
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

    if renderer is not None:
        renderer.set_thinking(current)

    # Agent chooses its best action — guarded so a crash doesn't kill pygame
    try:
        action, r, c = agent.choose_action(world)
    except Exception as exc:
        print(f"  [WARN] Agent {current} choose_action raised: {exc} — skipping turn")
        action, r, c = None, -1, -1
    finally:
        if renderer is not None:
            renderer.set_thinking(None)

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
            if renderer is not None:
                fx_key = _get_action_fx_key(action)
                if fx_key is not None and r >= 0 and c >= 0:
                    renderer.trigger_action_fx(fx_key, c, r)
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

        # ── Parameter customisation screen ─────────────────────────────────────
        custom_params = run_params_screen(screen, clock, a_type, b_type)

        # ── Build agents ───────────────────────────────────────────────────────
        print(f"\nBuilding agents:  A={a_type}  B={b_type}")
        agent_a = build_agent(a_type, AGENT_A, custom_params)
        agent_b = build_agent(b_type, AGENT_B, custom_params)
        agents  = {AGENT_A: agent_a, AGENT_B: agent_b}
        print(f"  Agent A -> {agent_a.name}")
        print(f"  Agent B -> {agent_b.name}\n")

        pygame.display.set_caption(
            f"{TITLE}  |  A: {agent_a.name}  vs  B: {agent_b.name}"
        )

        # ── Game state ─────────────────────────────────────────────────────────
        world         = World(custom_params=custom_params)
        renderer      = Renderer(screen)
        game_over     = False
        end_reason    = None
        auto_play     = True
        auto_delay_ms = AUTO_PLAY_DELAY_MS
        last_auto_ms  = 0
        turn_executor = TurnExecutor()  # Background thread for turn execution

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
                    elif event.key == pygame.K_SPACE and not game_over and not turn_executor.is_running:
                        turn_executor.start(world, agents, renderer)

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
                    # Start a turn in background thread
                    turn_executor.start(world, agents, renderer)
                    last_auto_ms = now_ms

            # Check if background turn completed
            if turn_executor.is_done() and turn_executor.result is not None:
                game_over, end_reason = turn_executor.get_result()
                turn_executor.result = None  # Clear result
                if game_over:
                    auto_play = False

            # ── Draw ──────────────────────────────────────────────────────────
            if game_over:
                renderer.draw_game_over(world, world.get_winner())
            else:
                renderer.draw(world)
            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    main()
