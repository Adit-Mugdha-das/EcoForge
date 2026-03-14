# =============================================================================
# agents/eval.py — Shared evaluation and reward functions used by all agents
# =============================================================================
#
# evaluate_state(world, agent) → float
#   Scores a world state from one agent's perspective.
#   Used by Minimax (to pick max/min nodes) and Monte Carlo (rollout scoring).
#
# compute_reward(old_stability, new_stability, world, agent) → float
#   Delta-based reward used by Q-Learning after each action.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *


# =============================================================================
# Main evaluation function
# =============================================================================

def evaluate_state(world, agent: str) -> float:
    """
    Score the world state from `agent`'s perspective.
    Combines 9 weighted components drawn from the design document.
    Higher score = better for `agent`.

    Components:
      1. Planet stability          ×10.0  (shared benefit)
      2. Resource level score      × 8.0  (water + food + oxygen together)
      3. Temperature optimality    ×12.0  (highest weight — critical range)
      4. Point differential        × 5.0  (my score vs opponent)
      5. Collapse penalty          × 1.0  (heavy negative near collapse)
      6. Strategic asset value     × 1.5  (own forests/farms/reservoirs)
      7. Territory control         × 2.0  (cells owned vs opponent)
      8. Population health         × 1.0  (growing pop = thriving ecosystem)
      9. Eco advantage             × 0.3  (own eco vs opponent eco)
    """
    opponent = AGENT_B if agent == AGENT_A else AGENT_A

    # 1. Planet stability (0.0 → 1.0)
    stability_score = world.stability * 10.0

    # 2. Resource levels — each meter scored on how close it is to optimal
    resource_score = (
        _score_meter(world.water_level)
        + _score_meter(world.food)
        + _score_meter(world.oxygen)
    ) / 3.0 * 8.0

    # 3. Temperature optimality — most heavily weighted
    temp_score = _score_temperature(world.temperature) * 12.0

    # 4. Point differential (rewards pulling ahead of opponent)
    point_diff = (world.scores[agent] - world.scores[opponent]) * 5.0

    # 5. Resource collapse penalty (heavy penalty when any resource is critical)
    collapse_penalty = _collapse_penalty(world) * 1.0

    # 6. Strategic asset value (agent's own built structures)
    asset_score = _asset_value(world, agent) * 1.5

    # 7. Territory control (number of cells owned)
    my_cells  = len(world.get_agent_cells(agent))
    opp_cells = len(world.get_agent_cells(opponent))
    territory_score = (my_cells - opp_cells) * 2.0

    # 8. Population health — population IS the ecosystem self-regulator.
    # A large, growing population means food/oxygen/water are all balanced.
    # A collapsed population (pop≈5) means the ecosystem has broken down.
    pop_ratio = world.population / 100.0        # 1.0 at pop=100, 2.0 at max pop=200
    pop_score = min(pop_ratio, 1.5) * 6.0       # 0 → 9.0 as pop grows to 150+

    # 9. Eco advantage — being eco-rich lets you act while opponent may be stuck
    eco_diff = world.eco_points[agent] - world.eco_points[opponent]
    eco_score = eco_diff * 0.3

    return (
        stability_score
        + resource_score
        + temp_score
        + point_diff
        + collapse_penalty
        + asset_score
        + territory_score
        + pop_score
        + eco_score
    )


# =============================================================================
# Q-Learning reward function
# =============================================================================

def compute_reward(old_stability: float, world, agent: str) -> float:
    """
    Reward signal for Q-Learning after one action + simulation step.
    Combines:
      - Environmental change: new_stability - old_stability  (PDF: "environment improves or worsens")
      - Bonus for maintaining stable range
      - Penalty for collapse proximity
      - Small bonus for building strategic assets
    """
    new_stability = world.stability
    delta         = new_stability - old_stability

    # Core: reward improvement, punish decline (scaled for meaningful gradient)
    reward = delta * 20.0

    # Bonus for staying in healthy stability zone
    if new_stability >= STABILITY_HIGH:
        reward += 3.0
    elif new_stability >= STABILITY_MODERATE:
        reward += 1.0
    elif new_stability < STABILITY_LOW:
        reward -= 5.0

    # Large penalty for approaching collapse
    if new_stability < STABILITY_COLLAPSE:
        reward -= 20.0

    # Population health signal — growing pop means ecosystem is working
    if world.population >= 100:
        reward += 2.0
    elif world.population < 20:
        reward -= 8.0   # population crash = ecosystem failure

    # Small positive for owning assets (encourages building)
    reward += _asset_value(world, agent) * 0.1

    return reward


# =============================================================================
# Component scoring helpers
# =============================================================================

def _score_meter(val: float, low: float = METER_OPTIMAL_LOW,
                 high: float = METER_OPTIMAL_HIGH) -> float:
    """
    Score a resource meter on -0.1 to 1.0.
    Matches world.py _score_meter exactly:
      1.0  = in optimal range (50–80)
      0.0  = at val=95 (high side) or val=0 (low side)
      -0.1 = at val=100 — slight negative, oversaturation actively hurts
    """
    if low <= val <= high:
        return 1.0
    elif val < low:
        return max(0.0, val / low)
    else:
        # Steeper penalty on high side: 0 at val=95, -0.1 at val=100
        return max(-0.1, 1.0 - (val - high) / 15.0)


def _score_temperature(val: float) -> float:
    """
    Temperature score 0.0–1.0.
    Optimal range: 40–60. Falls off steeply outside.
    """
    if TEMP_OPTIMAL_MIN <= val <= TEMP_OPTIMAL_MAX:
        return 1.0
    elif val < TEMP_OPTIMAL_MIN:
        return max(0.0, val / TEMP_OPTIMAL_MIN)
    else:
        return max(0.0, 1.0 - (val - TEMP_OPTIMAL_MAX) / 40.0)


def _collapse_penalty(world) -> float:
    """
    Heavy negative score when any resource is dangerously low OR dangerously high.
    Both extremes are bad — agents must keep all meters in the balanced zone.
    """
    penalty = 0.0
    for val in (world.water_level, world.food, world.oxygen):
        # Too low — crisis
        if val < 20:
            penalty -= 100.0
        elif val < 40:
            penalty -= 20.0
        # Too high — oversaturation (flooding, spoilage, imbalance)
        # Tiers match simulate.py thresholds: water>85 = farm flooding
        elif val > 90:
            penalty -= 30.0
        elif val > 85:
            penalty -= 18.0    # farm flooding zone: farm_efficiency *= 0.6
        elif val > 80:
            penalty -= 8.0

    # Temperature extremes — both directions equally dangerous
    if world.temperature < 25 or world.temperature > 80:
        penalty -= 50.0
    elif world.temperature < 30 or world.temperature > 75:
        penalty -= 20.0
    elif world.temperature < 40 or world.temperature > 65:
        penalty -= 8.0

    # Population collapse — crashed pop means ecosystem is broken
    # (meters fill up fast with no consumption → runaway oversaturation)
    if world.population < 15:
        penalty -= 25.0
    elif world.population < 30:
        penalty -= 10.0

    return penalty


def _asset_value(world, agent: str) -> float:
    """
    Score agent's strategic assets — context-sensitive.
    Building something that oversaturates a meter is less valuable.
    From .tex:  forests ×2, farms ×3, reservoirs ×4
    """
    value = 0.0

    # Context multipliers — asset value drops once the meter it fills exceeds 60.
    # Agents should stop building water/food/oxygen infra well before hitting 100.
    water_mult = 1.0 if world.water_level < 60 else max(0.0, 1.0 - (world.water_level - 60) / 30)
    food_mult  = 1.0 if world.food < 60        else max(0.0, 1.0 - (world.food - 60) / 30)
    oxy_mult   = 1.0 if world.oxygen < 60      else max(0.0, 1.0 - (world.oxygen - 60) / 30)

    for _, cell in world.get_agent_cells(agent):
        if cell.terrain == TERRAIN_FOREST:
            forest_val = (3.0 + cell.forest_maturity * 0.5) * oxy_mult  # 3.0–4.5
            # Extra discount if planet is already cold — more forests make it worse
            if world.temperature < TEMP_OPTIMAL_MIN:
                forest_val *= 0.3
            value += forest_val

        elif cell.terrain == TERRAIN_FARM:
            farm_val = (3.0 + cell.crop_stage * 0.5) * food_mult
            # Building more farms when water is already critically low is
            # counterproductive — each farm drains 0.2 water/step.
            if world.water_level < 40:
                water_stress = max(0.2, world.water_level / 40.0)
                farm_val *= water_stress
            value += farm_val

        elif cell.terrain == TERRAIN_RESERVOIR:
            value += 4.0 * water_mult

        elif cell.terrain == TERRAIN_RIVER:
            value += 1.0 * water_mult

        elif cell.terrain == TERRAIN_SOLAR:
            # Solar generates eco income (+1/step) AND warms the planet (+0.4°/step).
            # Very valuable when cold (planet needs warming), harmful when already hot.
            if world.temperature < TEMP_OPTIMAL_MIN:
                value += 3.5   # planet is cold — solar warming is critical
            elif world.temperature > TEMP_OPTIMAL_MAX:
                value += 0.8   # planet is hot — more heat is harmful (eco income only)
            else:
                value += 2.0   # optimal range — neutral warming, stable eco income

    return value
