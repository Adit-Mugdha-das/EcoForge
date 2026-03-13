# =============================================================================
# engine/simulate.py — World simulation: water, crops, forests, global meters
# Called once after every agent action (so twice per full round)
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


def simulate(world):
    """
    Run one full simulation step.
    Order matters: water spreads first, then biological growth uses updated water,
    then global meters are recalculated from the updated grid state.
    """
    _spread_water(world)
    _grow_crops(world)
    _mature_forests(world)
    _update_global_meters(world)
    world._update_stability()
    _award_turn_scores(world)


# =============================================================================
# Water spreading
# =============================================================================

def _spread_water(world):
    """
    Simulate water diffusion across the grid.

    Rules:
    - River and Reservoir tiles are permanent water sources (water = 10.0).
    - Every other tile receives water equal to (max neighbor water) - WATER_DECAY.
    - Uses a snapshot of current water values so spread order doesn't matter.
    """
    WATER_DECAY = 1.5   # water lost per cell of distance from source

    # Snapshot current water values to avoid order-of-update bias
    snapshot = [
        [world.grid[r][c].water for c in range(GRID_WIDTH)]
        for r in range(GRID_HEIGHT)
    ]

    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            cell = world.grid[r][c]

            # Permanent water sources never decay
            if cell.terrain in (TERRAIN_RIVER, TERRAIN_RESERVOIR):
                cell.water = 10.0
                continue

            # Find the highest water value among 4-directional neighbors
            neighbor_max = 0.0
            for (nr, nc), _ in world.get_neighbors(r, c):
                if snapshot[nr][nc] > neighbor_max:
                    neighbor_max = snapshot[nr][nc]

            # Spread: receive water from the richest neighbor, minus decay
            cell.water = max(0.0, neighbor_max - WATER_DECAY)


# =============================================================================
# Crop growth
# =============================================================================

def _grow_crops(world):
    """
    Advance crop growth stages on all Farm tiles.

    Growth requires:
    - The farm cell (or an adjacent cell) has water >= 3.
    - Crop is not yet mature (stage < 3).

    Growth stages: 0=none → 1=sprout → 2=growing → 3=mature
    Each stage takes CROP_GROWTH_RATE simulation steps to advance.
    """
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            cell = world.grid[r][c]
            if cell.terrain != TERRAIN_FARM:
                continue
            if cell.crop_stage >= 3:
                continue  # already mature, waiting for harvest

            # Check water access (this cell or any neighbor)
            has_water = cell.water >= 3.0
            if not has_water:
                for _, neighbor in world.get_neighbors(r, c):
                    if neighbor.water >= 3.0:
                        has_water = True
                        break

            if has_water:
                cell.crop_timer += 1
                if cell.crop_timer >= CROP_GROWTH_RATE:
                    cell.crop_stage = min(3, cell.crop_stage + 1)
                    cell.crop_timer = 0
            # No water = no growth this step (crops don't die, just stall)


# =============================================================================
# Forest maturation
# =============================================================================

def _mature_forests(world):
    """
    Advance forest maturity on all Forest tiles.

    Maturity levels: 0=just planted → 1=young → 2=medium → 3=mature
    Each level takes FOREST_MATURATION_TURNS simulation steps.
    Mature forests produce more oxygen and cool temperature more effectively.
    """
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            cell = world.grid[r][c]
            if cell.terrain != TERRAIN_FOREST:
                continue
            if cell.forest_maturity >= 3:
                continue  # fully mature, no further growth

            cell.forest_timer += 1
            if cell.forest_timer >= FOREST_MATURATION_TURNS:
                cell.forest_maturity = min(3, cell.forest_maturity + 1)
                cell.forest_timer = 0


# =============================================================================
# Global planet meter updates
# =============================================================================

def _update_global_meters(world):
    """
    Recalculate the four global planet meters (Water, Food, Oxygen, Temperature)
    based on the current grid state.

    Each meter uses a delta approach: a base per-turn change is computed from
    the grid structures, then added to the current meter value and clamped 0-100.
    This makes the meters feel like a living system that responds gradually.
    """
    # ---- Count grid structures ----
    river_count      = 0
    reservoir_count  = 0
    farm_count       = 0
    mature_farm_count = 0
    solar_count      = 0
    forest_maturity_sum = 0   # sum of all forest maturity levels (for oxygen + cooling)
    water_coverage   = 0      # cells with water > 0 (for temperature)

    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            cell = world.grid[r][c]

            if cell.terrain == TERRAIN_RIVER:
                river_count += 1
            elif cell.terrain == TERRAIN_RESERVOIR:
                reservoir_count += 1
            elif cell.terrain == TERRAIN_FARM:
                farm_count += 1
                if cell.crop_stage == 3:
                    mature_farm_count += 1
            elif cell.terrain == TERRAIN_SOLAR:
                solar_count += 1
            elif cell.terrain == TERRAIN_FOREST:
                forest_maturity_sum += cell.forest_maturity

            if cell.water > 0:
                water_coverage += 1

    # ---- Water level delta ----
    # Sources: rivers and reservoirs replenish water
    # Sinks: farms consume water through irrigation
    # Passive: slight natural evaporation each turn
    water_delta = (
        river_count     * 0.5    # rivers slowly refill water meter
        + reservoir_count * 1.5  # reservoirs are stronger water sources
        - farm_count    * 0.3    # farms consume water (irrigation)
        - 0.3                    # natural passive evaporation
    )
    world.water_level = _clamp(world.water_level + water_delta)

    # ---- Food delta ----
    # Only mature farms (stage 3) produce food each turn
    # Passive: small consumption each turn (population eats)
    food_delta = (
        mature_farm_count * 2.5  # food production from mature crops
        - 1.0                    # population consumption per turn
    )
    world.food = _clamp(world.food + food_delta)

    # ---- Oxygen delta ----
    # Forests produce oxygen; production scales with maturity
    # Farms and solar consume a little oxygen (land conversion / industry)
    # Passive: small atmospheric drain each turn
    oxygen_delta = (
        forest_maturity_sum * 0.4  # each level of forest maturity produces oxygen
        - farm_count  * 0.1        # agriculture slightly reduces oxygen
        - solar_count * 0.1        # industrial panels
        - 0.3                      # passive atmospheric drain
    )
    world.oxygen = _clamp(world.oxygen + oxygen_delta)

    # ---- Temperature delta ----
    # Farms (land clearing) and solar raise temperature
    # Forests cool temperature; effectiveness scales with maturity
    # Water coverage (wet cells) has a slight cooling effect
    temp_delta = (
        farm_count   * 0.2             # agriculture warms planet
        + solar_count * 0.4            # solar panels generate heat
        - forest_maturity_sum * 0.15   # mature forests cool most
        - water_coverage * 0.05        # wetlands moderate temperature
    )
    world.temperature = _clamp(world.temperature + temp_delta)


# =============================================================================
# Score awarding
# =============================================================================

def _award_turn_scores(world):
    """
    Award points to both agents each simulation step.

    Both agents share the planet — its stability benefits both, but the agent
    who built more valuable structures earns more individual points.
    """
    # Shared: both agents gain points proportional to planet stability
    stability_reward = world.stability * 1.0
    world.scores[AGENT_A] += stability_reward
    world.scores[AGENT_B] += stability_reward

    # Individual: points from strategic assets each agent owns
    for agent in (AGENT_A, AGENT_B):
        asset_score = 0.0
        for (r, c), cell in world.get_agent_cells(agent):
            if cell.terrain == TERRAIN_FOREST:
                asset_score += cell.forest_maturity * 0.2    # mature forests = more
            elif cell.terrain == TERRAIN_FARM:
                asset_score += cell.crop_stage * 0.15        # productive farms = more
            elif cell.terrain == TERRAIN_RESERVOIR:
                asset_score += 0.3                           # flat bonus
            elif cell.terrain == TERRAIN_RIVER:
                asset_score += 0.1
        world.scores[agent] += asset_score

    # Eco point passive income (small amount each sim step)
    ECO_PASSIVE = 1
    world.eco_points[AGENT_A] = min(99, world.eco_points[AGENT_A] + ECO_PASSIVE)
    world.eco_points[AGENT_B] = min(99, world.eco_points[AGENT_B] + ECO_PASSIVE)


# =============================================================================
# Utility
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
