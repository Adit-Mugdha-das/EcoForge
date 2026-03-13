# =============================================================================
# render/renderer.py — Draws the grid and UI panel using pygame-ce
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from config import *


class Renderer:
    """
    Handles all drawing for World Balance Architects.
    Call renderer.draw(world) once per frame inside the game loop.
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen

        # Fonts (pygame-ce uses the same font API as pygame)
        self.font_sm = pygame.font.SysFont('Segoe UI', 13)
        self.font_md = pygame.font.SysFont('Segoe UI', 16)
        self.font_lg = pygame.font.SysFont('Segoe UI', 21, bold=True)
        self.font_xl = pygame.font.SysFont('Segoe UI', 26, bold=True)

        # Pre-build owner tint surfaces (semi-transparent overlays)
        # These are drawn on top of terrain tiles to show ownership
        self._owner_tint = {
            AGENT_A: self._make_tint_surface(AGENT_A_COLOR, alpha=55),
            AGENT_B: self._make_tint_surface(AGENT_B_COLOR, alpha=55),
        }

        # Ownership border colors
        self._owner_border = {
            AGENT_A: AGENT_A_COLOR,
            AGENT_B: AGENT_B_COLOR,
        }

    # -------------------------------------------------------------------------
    # Public draw entry point
    # -------------------------------------------------------------------------

    def draw(self, world):
        """Draw everything: grid tiles, overlays, grid lines, UI panel."""
        self.screen.fill((15, 18, 28))       # dark background
        self._draw_tiles(world)
        self._draw_owner_tints(world)
        self._draw_agent_spawns(world)
        self._draw_grid_lines()
        self._draw_ui_panel(world)

    # -------------------------------------------------------------------------
    # Grid drawing
    # -------------------------------------------------------------------------

    def _draw_tiles(self, world):
        """Draw each cell as a colored rectangle based on terrain type."""
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell   = world.grid[r][c]
                color  = TILE_COLORS.get(cell.terrain, (100, 100, 100))
                rect   = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, color, rect)

                # Draw crop stage indicator (small dots on farm tiles)
                if cell.terrain == TERRAIN_FARM and cell.crop_stage > 0:
                    self._draw_crop_indicator(rect, cell.crop_stage)

                # Draw forest maturity indicator (small circle on forest tiles)
                if cell.terrain == TERRAIN_FOREST and cell.forest_maturity > 0:
                    self._draw_forest_indicator(rect, cell.forest_maturity)

    def _draw_owner_tints(self, world):
        """Overlay a semi-transparent tint on tiles owned by each agent."""
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = world.grid[r][c]
                if cell.owner in self._owner_tint:
                    rect = self._cell_rect(r, c)
                    self.screen.blit(self._owner_tint[cell.owner], rect.topleft)
                    # Draw a 2px border in agent color
                    pygame.draw.rect(
                        self.screen,
                        self._owner_border[cell.owner],
                        rect,
                        width=2,
                    )

    def _draw_agent_spawns(self, world):
        """
        Draw a circle marker at each agent's spawn position.
        Agent A = red circle, Agent B = blue circle.
        Only shown when the spawn tile has not been built on yet (is still LAND).
        """
        spawn_map = {
            AGENT_A: (AGENT_A_SPAWN, AGENT_A_COLOR),
            AGENT_B: (AGENT_B_SPAWN, AGENT_B_COLOR),
        }
        for agent, ((row, col), color) in spawn_map.items():
            cell = world.grid[row][col]
            rect = self._cell_rect(row, col)
            cx   = rect.centerx
            cy   = rect.centery
            # Draw outer ring
            pygame.draw.circle(self.screen, color, (cx, cy), TILE_SIZE // 3, width=3)
            # Draw agent letter inside
            label = self.font_lg.render(agent, True, color)
            self.screen.blit(label, label.get_rect(center=(cx, cy)))

    def _draw_grid_lines(self):
        """Draw thin lines separating each tile."""
        color = GRID_LINE_COLOR
        # Vertical lines
        for c in range(GRID_WIDTH + 1):
            x = c * TILE_SIZE
            pygame.draw.line(self.screen, color, (x, 0), (x, GRID_PIXEL_HEIGHT))
        # Horizontal lines
        for r in range(GRID_HEIGHT + 1):
            y = r * TILE_SIZE
            pygame.draw.line(self.screen, color, (0, y), (GRID_PIXEL_WIDTH, y))

    # -------------------------------------------------------------------------
    # Tile detail indicators
    # -------------------------------------------------------------------------

    def _draw_crop_indicator(self, tile_rect: pygame.Rect, stage: int):
        """Draw small white dots to indicate crop growth stage (1-3)."""
        dot_r = 4
        spacing = 12
        total_w = stage * spacing - (spacing - dot_r * 2)
        start_x = tile_rect.centerx - total_w // 2
        y       = tile_rect.bottom - 10
        for i in range(stage):
            cx = start_x + i * spacing
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, y), dot_r)

    def _draw_forest_indicator(self, tile_rect: pygame.Rect, maturity: int):
        """Draw a small circle whose size reflects forest maturity (1-3)."""
        radius = 5 + maturity * 3   # 8, 11, or 14 px
        pygame.draw.circle(
            self.screen,
            (20, 80, 20),
            tile_rect.center,
            radius,
        )

    # -------------------------------------------------------------------------
    # UI Panel
    # -------------------------------------------------------------------------

    def _draw_ui_panel(self, world):
        """Draw the right-side UI panel: meters, scores, turn, action log."""
        panel_rect = pygame.Rect(UI_PANEL_X, 0, UI_PANEL_WIDTH, SCREEN_HEIGHT)

        # Panel background
        pygame.draw.rect(self.screen, UI_BG_COLOR, panel_rect)
        # Panel left border
        pygame.draw.line(
            self.screen, UI_BORDER_COLOR,
            (UI_PANEL_X, 0), (UI_PANEL_X, SCREEN_HEIGHT), 2,
        )

        x  = UI_PANEL_X + 14
        y  = 14
        pw = UI_PANEL_WIDTH - 28   # usable width inside padding

        # ---- Title ----
        title = self.font_xl.render("World Balance", True, (200, 220, 255))
        self.screen.blit(title, (x, y))
        y += 30
        sub = self.font_sm.render("Architects", True, (130, 150, 190))
        self.screen.blit(sub, (x, y))
        y += 28

        # Divider
        pygame.draw.line(self.screen, UI_BORDER_COLOR, (x, y), (x + pw, y))
        y += 10

        # ---- Planet Meters ----
        meters = [
            ('WATER',   world.water_level,  METER_COLORS['water']),
            ('FOOD',    world.food,          METER_COLORS['food']),
            ('OXYGEN',  world.oxygen,        METER_COLORS['oxygen']),
            ('TEMP',    world.temperature,   METER_COLORS['temp']),
        ]
        for label, value, color in meters:
            y = self._draw_meter(x, y, pw, label, value, color)
            y += 6

        # Divider
        pygame.draw.line(self.screen, UI_BORDER_COLOR, (x, y), (x + pw, y))
        y += 10

        # ---- Stability ----
        stab_color = self._stability_color(world.stability)
        stab_label = self.font_md.render(
            f"Stability:  {world.stability:.2f}", True, stab_color
        )
        self.screen.blit(stab_label, (x, y))
        y += 24

        # Stability bar (full width)
        bar_rect = pygame.Rect(x, y, pw, 10)
        pygame.draw.rect(self.screen, (40, 40, 60), bar_rect, border_radius=5)
        fill_w = int(pw * world.stability)
        if fill_w > 0:
            pygame.draw.rect(
                self.screen, stab_color,
                pygame.Rect(x, y, fill_w, 10),
                border_radius=5,
            )
        y += 20

        # ---- Population ----
        pop_color = (140, 210, 160)   # soft green
        pop_txt = self.font_sm.render(
            f"Population:  {world.population:.0f}", True, pop_color
        )
        self.screen.blit(pop_txt, (x, y))
        y += 18

        # Divider
        pygame.draw.line(self.screen, UI_BORDER_COLOR, (x, y), (x + pw, y))
        y += 10

        # ---- Turn + Current Agent ----
        turn_txt = self.font_md.render(
            f"Turn  {world.turn} / {MAX_TURNS}", True, UI_TEXT_COLOR
        )
        self.screen.blit(turn_txt, (x, y))
        y += 22

        agent_color = AGENT_A_COLOR if world.current_agent == AGENT_A else AGENT_B_COLOR
        agent_txt = self.font_md.render(
            f"Active:  Agent {world.current_agent}", True, agent_color
        )
        self.screen.blit(agent_txt, (x, y))
        y += 26

        # Divider
        pygame.draw.line(self.screen, UI_BORDER_COLOR, (x, y), (x + pw, y))
        y += 10

        # ---- Scores ----
        scores_title = self.font_md.render("SCORES", True, (160, 170, 200))
        self.screen.blit(scores_title, (x, y))
        y += 22

        score_a = self.font_md.render(
            f"Agent A:  {world.scores[AGENT_A]:.1f}", True, AGENT_A_COLOR
        )
        score_b = self.font_md.render(
            f"Agent B:  {world.scores[AGENT_B]:.1f}", True, AGENT_B_COLOR
        )
        self.screen.blit(score_a, (x, y));       y += 20
        self.screen.blit(score_b, (x, y));       y += 24

        # Eco points
        eco_a = self.font_sm.render(
            f"Eco pts A:  {world.eco_points[AGENT_A]}", True, (180, 180, 180)
        )
        eco_b = self.font_sm.render(
            f"Eco pts B:  {world.eco_points[AGENT_B]}", True, (180, 180, 180)
        )
        self.screen.blit(eco_a, (x, y));  y += 18
        self.screen.blit(eco_b, (x, y));  y += 22

        # Divider
        pygame.draw.line(self.screen, UI_BORDER_COLOR, (x, y), (x + pw, y))
        y += 10

        # ---- Action Log ----
        log_title = self.font_md.render("ACTION LOG", True, (160, 170, 200))
        self.screen.blit(log_title, (x, y))
        y += 20

        for entry in world.action_log[-5:]:
            log_line = self.font_sm.render(entry, True, (160, 170, 190))
            self.screen.blit(log_line, (x, y))
            y += 16

    # -------------------------------------------------------------------------
    # Meter bar helper
    # -------------------------------------------------------------------------

    def _draw_meter(
        self,
        x: int, y: int, width: int,
        label: str, value: float, color: tuple,
    ) -> int:
        """
        Draw a labeled meter bar.
        Returns the new y position after drawing.
        """
        # Label + numeric value
        bar_label = self.font_sm.render(
            f"{label}  {value:.1f}", True, UI_TEXT_COLOR
        )
        self.screen.blit(bar_label, (x, y))
        y += 16

        # Bar background
        bar_h    = 10
        bar_rect = pygame.Rect(x, y, width, bar_h)
        pygame.draw.rect(self.screen, (40, 45, 60), bar_rect, border_radius=4)

        # Bar fill (clamped 0-100)
        fill_ratio = max(0.0, min(1.0, value / 100.0))
        fill_w     = int(width * fill_ratio)
        if fill_w > 0:
            pygame.draw.rect(
                self.screen, color,
                pygame.Rect(x, y, fill_w, bar_h),
                border_radius=4,
            )

        return y + bar_h

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Return the screen Rect for a grid cell."""
        return pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)

    def _make_tint_surface(self, color: tuple, alpha: int) -> pygame.Surface:
        """Create a transparent colored surface for ownership tinting."""
        surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        surf.fill((*color, alpha))
        return surf

    def _stability_color(self, stability: float) -> tuple:
        """Return a color representing the current stability level."""
        if stability >= STABILITY_HIGH:
            return (80, 220, 120)    # green — healthy
        elif stability >= STABILITY_MODERATE:
            return (220, 200, 60)    # yellow — moderate
        elif stability >= STABILITY_LOW:
            return (220, 130, 40)    # orange — stressed
        else:
            return (220, 60, 60)     # red — critical
