# =============================================================================
# engine/actions.py — All 7 agent actions
# Each action class provides:
#   .name           human-readable name
#   .cost           eco points required
#   .get_valid_targets(world, agent) → list of (row, col)
#   .apply(world, agent, row, col)  → mutates world, returns log string
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


# =============================================================================
# Base class
# =============================================================================

class Action:
    name: str = "base"
    cost: int = 0

    def can_afford(self, world, agent) -> bool:
        return world.eco_points[agent] >= self.cost

    def get_valid_targets(self, world, agent) -> list:
        """Return list of (row, col) where this action can legally be applied."""
        raise NotImplementedError

    def apply(self, world, agent, row: int, col: int) -> str:
        """
        Mutate the world state (place structure, deduct eco points, log).
        Returns a short log string describing what happened.
        """
        raise NotImplementedError

    def _deduct(self, world, agent):
        world.eco_points[agent] -= self.cost

    def _mark_owner(self, cell, agent):
        """Assign ownership when an agent builds on a cell."""
        cell.owner = agent


# =============================================================================
# 1. Build Canal  (Land → River)
# =============================================================================

class BuildCanal(Action):
    """
    Convert a Land tile to a River tile.
    The target must be adjacent to an existing water source (river or reservoir).
    This extends the water network and raises the Water metre over time.
    """
    name = "Build Canal"
    cost = ACTION_COSTS['build_canal']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        targets = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = world.grid[r][c]
                if cell.terrain != TERRAIN_LAND:
                    continue
                # Must be adjacent to a water source
                for _, neighbor in world.get_neighbors(r, c):
                    if neighbor.terrain in (TERRAIN_RIVER, TERRAIN_RESERVOIR):
                        targets.append((r, c))
                        break
        return targets

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain = TERRAIN_RIVER
        cell.water   = 10.0
        self._mark_owner(cell, agent)
        self._deduct(world, agent)
        return f"Agent {agent}: Built canal at ({row},{col})"


# =============================================================================
# 2. Place Reservoir  (Land → Reservoir)
# =============================================================================

class BuildReservoir(Action):
    """
    Build a water reservoir on a Land tile adjacent to a water source.
    Reservoirs are stronger water amplifiers than canals (water = 10, radius +2).
    """
    name = "Build Reservoir"
    cost = ACTION_COSTS['place_reservoir']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        targets = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = world.grid[r][c]
                if cell.terrain != TERRAIN_LAND:
                    continue
                for _, neighbor in world.get_neighbors(r, c):
                    if neighbor.terrain in (TERRAIN_RIVER, TERRAIN_RESERVOIR):
                        targets.append((r, c))
                        break
        return targets

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain = TERRAIN_RESERVOIR
        cell.water   = 10.0
        self._mark_owner(cell, agent)
        self._deduct(world, agent)
        return f"Agent {agent}: Built reservoir at ({row},{col})"


# =============================================================================
# 3. Plant Forest  (Land → Forest, maturity 0)
# =============================================================================

class PlantForest(Action):
    """
    Plant a forest on any Land tile.
    Forests start at maturity 0 and grow to maturity 3 over FOREST_MATURATION_TURNS steps.
    Mature forests produce oxygen and cool temperature.
    """
    name = "Plant Forest"
    cost = ACTION_COSTS['plant_forest']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        return [
            (r, c)
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
            if world.grid[r][c].terrain == TERRAIN_LAND
        ]

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain         = TERRAIN_FOREST
        cell.forest_maturity = 0
        cell.forest_timer    = 0
        self._mark_owner(cell, agent)
        self._deduct(world, agent)
        return f"Agent {agent}: Planted forest at ({row},{col})"


# =============================================================================
# 4. Clear Forest  (Forest → Land)
# =============================================================================

class ClearForest(Action):
    """
    Remove a forest tile, reverting it to Land.
    Cheap but harmful: reduces oxygen and raises temperature.
    An agent can only clear forests they own.
    """
    name = "Clear Forest"
    cost = ACTION_COSTS['clear_forest']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        return [
            (r, c)
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
            if (world.grid[r][c].terrain == TERRAIN_FOREST
                and world.grid[r][c].owner == agent)
        ]

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain         = TERRAIN_LAND
        cell.forest_maturity = 0
        cell.forest_timer    = 0
        cell.owner           = None    # cleared land reverts to neutral
        self._deduct(world, agent)
        return f"Agent {agent}: Cleared forest at ({row},{col})"


# =============================================================================
# 5. Plant Crop  (Land → Farm, stage 0)
# =============================================================================

class PlantFarm(Action):
    """
    Establish a farm on a Land tile that has water access.
    Water access means: the cell or any direct neighbor has water >= 3.
    Crops grow over several simulation steps before they can be harvested.
    """
    name = "Plant Farm"
    cost = ACTION_COSTS['plant_crop']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        targets = []
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                cell = world.grid[r][c]
                if cell.terrain != TERRAIN_LAND:
                    continue
                # Needs water access
                has_water = cell.water >= 3.0
                if not has_water:
                    for _, neighbor in world.get_neighbors(r, c):
                        if neighbor.water >= 3.0:
                            has_water = True
                            break
                if has_water:
                    targets.append((r, c))
        return targets

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain     = TERRAIN_FARM
        cell.crop_stage  = 0
        cell.crop_timer  = 0
        self._mark_owner(cell, agent)
        self._deduct(world, agent)
        return f"Agent {agent}: Planted farm at ({row},{col})"


# =============================================================================
# 6. Harvest Crop  (Farm stage 3 → Farm stage 0, food burst)
# =============================================================================

class HarvestCrop(Action):
    """
    Harvest a mature (stage 3) farm owned by this agent.
    Gives an immediate food boost and resets the crop to stage 0 to regrow.
    """
    name = "Harvest Crop"
    cost = ACTION_COSTS['harvest_crop']

    FOOD_YIELD = 15.0   # immediate food meter gain on harvest (main food income source)

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        return [
            (r, c)
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
            if (world.grid[r][c].terrain == TERRAIN_FARM
                and world.grid[r][c].crop_stage == 3
                and world.grid[r][c].owner == agent)
        ]

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.crop_stage = 0   # reset to regrow
        cell.crop_timer = 0
        # Harvested food is capped by storage — only add what fits below 85
        # (realistic: surplus crops spoil if silos are full)
        actual_yield = max(0.0, min(self.FOOD_YIELD, 85.0 - world.food))
        world.food = min(100.0, world.food + actual_yield)
        self._deduct(world, agent)
        # Score bonus regardless — agent did the work
        world.scores[agent] += 2.0
        return (f"Agent {agent}: Harvested crop at ({row},{col}) "
                f"[+{actual_yield:.0f} food]")


# =============================================================================
# 7. Build Solar Plant  (Land → Solar)
# =============================================================================

class BuildSolarPlant(Action):
    """
    Build a solar energy plant on any Land tile.
    Expensive and warms the planet, but earns eco-point income faster and
    contributes to strategic asset scoring. Use with caution near temperature
    limits.
    """
    name = "Build Solar Plant"
    cost = ACTION_COSTS['build_solar']

    def get_valid_targets(self, world, agent) -> list:
        if not self.can_afford(world, agent):
            return []
        return [
            (r, c)
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
            if world.grid[r][c].terrain == TERRAIN_LAND
        ]

    def apply(self, world, agent, row, col) -> str:
        cell = world.grid[row][col]
        cell.terrain = TERRAIN_SOLAR
        self._mark_owner(cell, agent)
        self._deduct(world, agent)
        return f"Agent {agent}: Built solar plant at ({row},{col})"


# =============================================================================
# 8. Adjust Resource Allocation  (no tile change — sets agent priority mode)
# =============================================================================

class AdjustAllocation(Action):
    """
    Set this agent's priority mode: 'farm', 'forest', or 'balanced'.
    This is a free action (0 eco points) that influences AI decision-making
    and can subtly affect simulation weights for this agent's owned tiles.
    row and col are ignored — pass (0, 0) or (-1, -1).
    """
    name = "Adjust Resource Allocation"
    cost = ACTION_COSTS['adjust_allocation']

    MODES = ('farm', 'forest', 'balanced')

    def __init__(self, mode: str = 'balanced'):
        assert mode in self.MODES, f"Invalid mode: {mode}"
        self.mode = mode

    def get_valid_targets(self, world, agent) -> list:
        # Always valid, target is irrelevant — return a sentinel list
        return [(-1, -1)]

    def apply(self, world, agent, row, col) -> str:
        world.allocation_mode[agent] = self.mode
        return f"Agent {agent}: Set allocation -> {self.mode}"


# =============================================================================
# Registry and convenience helpers
# =============================================================================

# All 7 actions — names match PDF exactly
# Tile actions require a (row, col) target
TILE_ACTIONS = [
    PlantForest(),
    BuildCanal(),
    BuildReservoir(),
    PlantFarm(),
    HarvestCrop(),
    ClearForest(),
    BuildSolarPlant(),
]

# All allocation modes as pre-built action objects (free action, no tile target)
ALLOCATION_ACTIONS = [AdjustAllocation(m) for m in AdjustAllocation.MODES]

# Complete action list (for reference)
ALL_ACTIONS = TILE_ACTIONS + ALLOCATION_ACTIONS


def get_all_valid_moves(world, agent) -> list:
    """
    Return every legal (action, row, col) tuple for the given agent
    in the current world state. Used by AI agents to enumerate choices.
    """
    moves = []
    for action in TILE_ACTIONS:
        for (r, c) in action.get_valid_targets(world, agent):
            moves.append((action, r, c))
    for action in ALLOCATION_ACTIONS:
        moves.append((action, -1, -1))
    return moves


def apply_action(world, agent, action: Action, row: int, col: int) -> str:
    """
    Safely apply an action. Checks eco points first.
    Returns a log string (success or failure reason).
    """
    if not action.can_afford(world, agent):
        return f"Agent {agent}: Cannot afford {action.name} (need {action.cost} eco pts)"
    log = action.apply(world, agent, row, col)
    world.log_action(log)
    return log
