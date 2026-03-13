# =============================================================================
# engine/world.py — Cell and World classes; core grid state
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


class Cell:
    """
    Represents a single tile on the 10x10 grid.
    Holds terrain type, ownership, and all per-cell resource values.
    """

    def __init__(self):
        self.terrain          = TERRAIN_LAND  # land/river/farm/forest/reservoir/solar
        self.owner            = None          # AGENT_A, AGENT_B, or None (neutral)
        self.water            = 0.0           # local water level 0-10
        self.crop_stage       = 0             # 0=none, 1=sprout, 2=growing, 3=mature
        self.crop_timer       = 0             # turns since last growth stage
        self.forest_maturity  = 0             # 0=none, 1=young, 2=mid, 3=mature
        self.forest_timer     = 0             # turns since planted/last maturity gain
        self.is_agent_spawn   = None          # marks starting position: AGENT_A/AGENT_B

    def is_buildable(self):
        """Can an agent place a new structure here?"""
        return self.terrain == TERRAIN_LAND

    def has_water_access(self, world, row, col):
        """
        Check if this cell or any 4-directional neighbor has water.
        Needed before planting a crop (requires water >= 3 nearby).
        """
        if self.water >= 3:
            return True
        for _, neighbor in world.get_neighbors(row, col):
            if neighbor.water >= 3:
                return True
        return False

    def __repr__(self):
        return (f"Cell(terrain={self.terrain}, owner={self.owner}, "
                f"water={self.water:.1f}, crop={self.crop_stage}, "
                f"forest={self.forest_maturity})")


class World:
    """
    The full game state: 10x10 grid + global planet meters + turn tracking.
    All simulation and action results flow through this object.
    """

    def __init__(self):
        # Build the grid
        self.grid = [[Cell() for _ in range(GRID_WIDTH)]
                     for _ in range(GRID_HEIGHT)]

        # Global planet meters (each 0-100)
        self.water_level  = INITIAL_WATER
        self.food         = INITIAL_FOOD
        self.oxygen       = INITIAL_OXYGEN
        self.temperature  = INITIAL_TEMPERATURE

        # Planet stability index (0.0-1.0), recomputed each simulation step
        self.stability    = 0.0

        # Turn tracking
        self.turn          = 0               # increments after BOTH agents have acted
        self.half_turn     = 0               # increments after each agent acts (A=even, B=odd)
        self.current_agent = AGENT_A         # whose turn it is

        # Per-agent eco points (action currency)
        self.eco_points = {
            AGENT_A: STARTING_ECO_POINTS,
            AGENT_B: STARTING_ECO_POINTS,
        }

        # Per-agent scores (accumulated over the game)
        self.scores = {
            AGENT_A: 0.0,
            AGENT_B: 0.0,
        }

        # Per-agent allocation priority mode: 'farm', 'forest', or 'balanced'
        self.allocation_mode = {
            AGENT_A: 'balanced',
            AGENT_B: 'balanced',
        }

        # Action log for the UI panel (last 5 actions)
        self.action_log = []

        # Collapse counter (consecutive turns at critical stability)
        self.collapse_turns = 0

        # Apply starting layout
        self._setup_initial_grid()

        # Compute initial stability score
        self._update_stability()

    # -------------------------------------------------------------------------
    # Grid setup
    # -------------------------------------------------------------------------

    def _setup_initial_grid(self):
        """
        Place initial river tiles and mark agent spawn positions.

        Layout (10x10):
            . . . . . . . . . .
            . A . . . . . . . .
            . . . . . . . . . .
            . . . . ~ . . . . .
            . . . . ~ . . . . .
            . . . . ~ ~ . . . .
            . . . . . ~ . . . .
            . . . . . . . . . .
            . . . . . . . . B .
            . . . . . . . . . .
        """
        # River tiles (L-shape through center)
        for (row, col) in INITIAL_RIVER_TILES:
            cell = self.grid[row][col]
            cell.terrain = TERRAIN_RIVER
            cell.water   = 10.0

        # Mark agent spawn positions (still LAND — just flagged)
        r_a, c_a = AGENT_A_SPAWN
        r_b, c_b = AGENT_B_SPAWN
        self.grid[r_a][c_a].is_agent_spawn = AGENT_A
        self.grid[r_b][c_b].is_agent_spawn = AGENT_B

    # -------------------------------------------------------------------------
    # Grid accessors
    # -------------------------------------------------------------------------

    def get_cell(self, row, col):
        """Return the Cell at (row, col), or None if out of bounds."""
        if 0 <= row < GRID_HEIGHT and 0 <= col < GRID_WIDTH:
            return self.grid[row][col]
        return None

    def get_neighbors(self, row, col, diagonal=False):
        """
        Return a list of ((row, col), Cell) tuples for valid neighbors.
        4-directional by default; pass diagonal=True for 8-directional.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        result = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            cell = self.get_cell(nr, nc)
            if cell is not None:
                result.append(((nr, nc), cell))
        return result

    def get_all_cells_of_type(self, terrain_type):
        """Return list of ((row, col), Cell) for all cells matching terrain_type."""
        result = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = self.grid[r][c]
                if cell.terrain == terrain_type:
                    result.append(((r, c), cell))
        return result

    def get_agent_cells(self, agent):
        """Return all cells owned by the given agent."""
        result = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = self.grid[r][c]
                if cell.owner == agent:
                    result.append(((r, c), cell))
        return result

    # -------------------------------------------------------------------------
    # Stability calculation
    # -------------------------------------------------------------------------

    def _score_meter(self, val, low=METER_OPTIMAL_LOW, high=METER_OPTIMAL_HIGH):
        """
        Score a resource meter on 0.0-1.0.
        1.0 when in optimal range (50-80), degrades linearly outside.
        """
        if low <= val <= high:
            return 1.0
        elif val < low:
            return max(0.0, val / low)
        else:
            return max(0.0, 1.0 - (val - high) / 20.0)

    def _score_temperature(self, val):
        """
        Score temperature on 0.0-1.0.
        Optimal: 40-60. Degrades toward 0 as it approaches 0 or 100.
        """
        if TEMP_OPTIMAL_MIN <= val <= TEMP_OPTIMAL_MAX:
            return 1.0
        elif val < TEMP_OPTIMAL_MIN:
            return max(0.0, val / TEMP_OPTIMAL_MIN)
        else:
            return max(0.0, 1.0 - (val - TEMP_OPTIMAL_MAX) / 40.0)

    def _update_stability(self):
        """
        Recompute planet stability as an equal-weighted average of all 4 meter scores.
        Clamps result to [0.0, 1.0].
        """
        w = self._score_meter(self.water_level)
        f = self._score_meter(self.food)
        o = self._score_meter(self.oxygen)
        t = self._score_temperature(self.temperature)
        self.stability = max(0.0, min(1.0, 0.25 * (w + f + o + t)))

    # -------------------------------------------------------------------------
    # Turn management
    # -------------------------------------------------------------------------

    def switch_agent(self):
        """
        End the current agent's turn and pass control to the other.
        Increments the full turn counter after both agents have acted.
        """
        if self.current_agent == AGENT_A:
            self.current_agent = AGENT_B
        else:
            self.current_agent = AGENT_A
            self.turn += 1   # full round complete

        self.half_turn += 1

    # -------------------------------------------------------------------------
    # Action log
    # -------------------------------------------------------------------------

    def log_action(self, message):
        """Append a message to the action log (capped at 5 entries)."""
        self.action_log.append(message)
        if len(self.action_log) > 5:
            self.action_log.pop(0)

    # -------------------------------------------------------------------------
    # State abstraction (Q-Learning)
    # -------------------------------------------------------------------------

    def get_state_category(self):
        """
        Return an abstracted state tuple for Q-Learning.
        Each of the 4 meters is bucketed into: 0=low, 1=medium, 2=high.
        Total: 3^4 = 81 possible states.
        Returns: (water_cat, food_cat, oxygen_cat, temp_cat)
        """
        def cat(val):
            if val < LOW_THRESHOLD:  return 0
            if val < HIGH_THRESHOLD: return 1
            return 2

        def cat_temp(val):
            # Temperature: optimal range (40-60) = medium
            if val < TEMP_OPTIMAL_MIN:  return 0   # too cold
            if val <= TEMP_OPTIMAL_MAX: return 1   # optimal
            return 2                                # too hot

        return (
            cat(self.water_level),
            cat(self.food),
            cat(self.oxygen),
            cat_temp(self.temperature),
        )

    # -------------------------------------------------------------------------
    # Win / loss conditions
    # -------------------------------------------------------------------------

    def is_game_over(self):
        """
        Check whether the game has ended.
        Returns: (bool, reason_string or None)
        Reasons: 'collapse', 'timeout'
        """
        if self.stability < STABILITY_COLLAPSE:
            self.collapse_turns += 1
        else:
            self.collapse_turns = 0

        if self.collapse_turns >= 3:
            return True, 'collapse'

        if self.turn >= MAX_TURNS:
            return True, 'timeout'

        return False, None

    def get_winner(self):
        """
        Return the agent with the higher score, or None on a draw.
        """
        if self.scores[AGENT_A] > self.scores[AGENT_B]:
            return AGENT_A
        elif self.scores[AGENT_B] > self.scores[AGENT_A]:
            return AGENT_B
        return None  # draw

    # -------------------------------------------------------------------------
    # Debug helper
    # -------------------------------------------------------------------------

    def print_grid(self):
        """Print a basic ASCII representation of the grid to the terminal."""
        symbols = {
            TERRAIN_LAND:      '.',
            TERRAIN_RIVER:     '~',
            TERRAIN_FARM:      'F',
            TERRAIN_FOREST:    'T',
            TERRAIN_RESERVOIR: 'R',
            TERRAIN_SOLAR:     'S',
        }
        print(f"\n--- Turn {self.turn} | Agent {self.current_agent}'s turn ---")
        for r in range(GRID_HEIGHT):
            row_str = ''
            for c in range(GRID_WIDTH):
                cell = self.grid[r][c]
                if cell.is_agent_spawn == AGENT_A:
                    row_str += 'A '
                elif cell.is_agent_spawn == AGENT_B:
                    row_str += 'B '
                else:
                    row_str += symbols.get(cell.terrain, '?') + ' '
            print(row_str)
        print(f"Water:{self.water_level:.1f}  Food:{self.food:.1f}  "
              f"Oxygen:{self.oxygen:.1f}  Temp:{self.temperature:.1f}  "
              f"Stability:{self.stability:.3f}")
