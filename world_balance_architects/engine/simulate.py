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
    Recalculate the four global planet meters (Water, Food, Oxygen, Temperature).

    Each meter uses a delta approach. Critically, meters interact:
      - Too much water floods farms (less food) and evaporates faster
      - Too much food rots (spoilage drain)
      - Too much oxygen raises temperature (oxidation heat)
      - High temperature evaporates water and stresses crops
    This creates realistic push-and-pull so agents must keep ALL meters balanced.
    """
    # ---- Count grid structures ----
    river_count         = 0
    reservoir_count     = 0
    farm_count          = 0
    mature_farm_count   = 0
    solar_count         = 0
    forest_count        = 0
    forest_maturity_sum = 0
    water_coverage      = 0

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
                forest_count += 1
                forest_maturity_sum += cell.forest_maturity
            if cell.water > 0:
                water_coverage += 1

    # ---- Derived cross-meter effects ----

    # Flooding: excess water (>80) drains strongly — runoff + evaporation
    flood_drain = max(0.0, world.water_level - 80) * 0.15

    # Heat evaporation: high temperature dries out the planet
    heat_evap = max(0.0, world.temperature - 70) * 0.06

    # Food spoilage: food starts rotting at 72 (8 pts inside optimal), not at 80.
    # This prevents food from spiking to 85+ when population crashes and farm
    # production far outpaces the reduced consumption floor.
    food_spoilage = max(0.0, world.food - 72) * 0.30

    # Oxygen self-regulation: excess oxygen (>80) dissipates faster
    # Rate raised 0.40→0.65: at oxygen=82 with 10 forests bleed now exceeds production
    oxygen_bleed = max(0.0, world.oxygen - 80) * 0.65

    # Excess oxygen causes atmospheric oxidation → slight warming
    oxygen_heat = max(0.0, world.oxygen - 80) * 0.04

    # Heat radiation: above 80°, planet radiates excess energy to space.
    # Equivalent to oxygen_bleed / food_spoilage — temperature finally has self-regulation.
    # Coefficient 0.175 → equilibrium at temp≈92 with 3 solar + 4 farms + no forests.
    # Forests pull it much lower; this is a safety net, not a replacement for forests.
    heat_radiation = max(0.0, world.temperature - 80) * 0.175

    # Heat decomposition: high temperature breaks down organic matter → drains oxygen
    heat_decomp = max(0.0, world.temperature - 65) * 0.06

    # Wildfire atmospheric drain: when oxygen oversaturates above 85, organic fuel
    # ignites more easily — proportional to forest density (more fuel = more burn).
    # This is deterministic for agent lookahead (no random burns here).
    wildfire_drain = max(0.0, world.oxygen - 85) * forest_count * 0.04

    # ---- Population dynamics ----
    # Base growth is food-driven. Logistic scaling is applied ONLY to the growth
    # direction (positive pop_change): growth slows as population approaches the
    # carrying capacity (200). Crashes are left unscaled — high population with
    # scarce resources should crash just as fast regardless of density.
    base_growth = (world.food - 50) * 0.08
    # Ecosystem thriving bonus: when all four meters are simultaneously in optimal range,
    # the planet is healthy enough to support faster reproduction — realistic overpopulation
    # scenario when food, water, oxygen, and temperature are all ideal.
    # Added BEFORE the logistic brake so it also scales down near carrying capacity.
    if (METER_OPTIMAL_LOW  <= world.water_level  <= METER_OPTIMAL_HIGH and
            METER_OPTIMAL_LOW  <= world.food        <= METER_OPTIMAL_HIGH and
            METER_OPTIMAL_LOW  <= world.oxygen      <= METER_OPTIMAL_HIGH and
            TEMP_OPTIMAL_MIN   <= world.temperature <= TEMP_OPTIMAL_MAX):
        base_growth += 0.4
    if base_growth > 0:
        base_growth *= (1.0 - world.population / 200.0)   # logistic brake on growth
    pop_change = base_growth
    if world.oxygen < 25:
        pop_change -= 2.5    # suffocation (softened from -4.0 — early game is harsh enough)
    elif world.oxygen < 40:
        pop_change -= 0.8    # low oxygen stress (softened from -1.5)
    if world.temperature > 75:
        pop_change -= 2.0    # extreme heat
    elif world.temperature > 65:
        pop_change -= 0.8    # heat stress
    if world.water_level < 15:
        pop_change -= 1.5    # severe drought
    elif world.water_level < 30:
        pop_change -= 0.6    # mild water stress
    pop_change = max(-2.5, min(2.5, pop_change))   # symmetric cap
    world.population = max(5.0, min(200.0, world.population + pop_change))

    # Population consumes food, oxygen, and water proportionally to their size.
    # Oxygen breathing is also proportional to available oxygen — when oxygen is
    # scarce, organisms conserve it (less breathing in thin air = realistic).
    food_by_pop  = max(world.population * 0.04, 1.0)    # min 1.0/step
    oxy_ratio    = min(1.0, world.oxygen / 40.0)         # full breathing at oxy≥40, less below
    oxygen_by_pop = max(world.population * 0.04, 1.6) * oxy_ratio   # breathing floor scales with oxygen
    water_by_pop = max(world.population * 0.01, 0.25)   # min 0.25/step

    # Crop heat/flood stress
    if world.temperature > 75:
        farm_efficiency = 0.4    # severe heat stress
    elif world.temperature > 65:
        farm_efficiency = 0.7    # moderate heat stress
    else:
        farm_efficiency = 1.0    # normal production
    if world.water_level > 85:
        farm_efficiency *= 0.6   # flooded fields

    # ---- Water level delta ----
    # Forests transpire moisture back into the ecosystem (water cycle)
    forest_transpiration = forest_count * 0.05
    water_delta = (
        river_count          * 0.3    # rivers refill water (raised 0.2→0.3 so 5 rivers sustain planet)
        + reservoir_count    * 0.3    # reservoirs amplify water
        + forest_transpiration         # forests release moisture (water cycle)
        - farm_count         * 0.2    # farms need irrigation (lowered 0.4→0.2 — 0.4 was too draining)
        - water_by_pop                 # population drinks/uses water
        - 0.8                          # passive evaporation
        - flood_drain                  # strong drain when flooded (>80)
        - heat_evap                    # extra drain in extreme heat
    )
    world.water_level = _clamp(world.water_level + water_delta)

    # ---- Food delta ----
    food_delta = (
        mature_farm_count * 0.5 * farm_efficiency  # was 0.8 — reduced to prevent overproduction
        - food_by_pop                               # population eats — scales with effective_pop
        - food_spoilage                             # excess food rots (>80)
    )
    world.food = _clamp(world.food + food_delta)

    # ---- Oxygen delta ----
    oxygen_delta = (
        forest_count          * 0.2   # base oxygen from any forest
        + forest_maturity_sum * 0.08  # bonus from mature forests
        - farm_count          * 0.05  # agriculture slightly reduces oxygen
        - solar_count         * 0.1   # industrial panels consume oxygen
        - oxygen_by_pop               # population breathes — scales with population
        - 0.3                         # passive atmospheric drain
        - heat_decomp                 # high temperature → decomposition → O2 drain
        - oxygen_bleed                # excess oxygen dissipates (>80)
        - wildfire_drain              # high O2 + dense forests → combustion drain (>85)
    )
    world.oxygen = _clamp(world.oxygen + oxygen_delta)

    # ---- Temperature delta ----
    # Forest cooling is context-sensitive: strongest in extreme heat, zero when cold.
    # Above 80° an emergency tier kicks in — double the cooling rate — because
    # transpiration and albedo effects are much stronger under severe heat stress.
    if world.temperature > 80:
        forest_cooling = forest_maturity_sum * 0.20   # emergency: double rate above 80°
    elif world.temperature > TEMP_OPTIMAL_MAX:
        forest_cooling = forest_maturity_sum * 0.10   # active cooling in heat
    elif world.temperature < TEMP_OPTIMAL_MIN:
        forest_cooling = 0.0                          # no cooling when already cold
    else:
        forest_cooling = forest_maturity_sum * 0.05   # gentle damping in optimal range

    temp_delta = (
        + 0.5                    # passive greenhouse warming
        + farm_count   * 0.1     # agriculture warms planet
        + solar_count  * 0.4     # solar panels generate heat
        + oxygen_heat            # excess oxygen causes oxidative warming
        - forest_cooling         # context-sensitive forest cooling
        - water_coverage * 0.01  # wetlands cool slightly
        - heat_radiation         # planetary heat radiation — self-regulation above 80
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

    # Eco point income — base passive + solar bonus + mature farm bonus
    # Solar plants are energy generators — their main game role is eco supply
    for agent in (AGENT_A, AGENT_B):
        eco_gain = 2   # base passive income (raised from 1 to ensure agents can always act)
        for (r, c), cell in world.get_agent_cells(agent):
            if cell.terrain == TERRAIN_SOLAR:
                eco_gain += 1        # each solar plant adds +1 eco/step
            elif cell.terrain == TERRAIN_FARM and cell.crop_stage == 3:
                eco_gain += 0        # mature farms: income via harvest only
        world.eco_points[agent] = min(99, world.eco_points[agent] + eco_gain)


# =============================================================================
# Utility
# =============================================================================

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
