# =============================================================================
# config.py — All global constants for World Balance Architects
# =============================================================================

# --- Screen ---
SCREEN_WIDTH  = 900   # 600px grid + 300px UI panel
SCREEN_HEIGHT = 640
FPS           = 60
TITLE         = "World Balance Architects"

# --- Grid ---
GRID_WIDTH    = 10
GRID_HEIGHT   = 10
TILE_SIZE     = 60    # pixels per tile (10 * 60 = 600px for grid)

# --- Derived layout values ---
GRID_PIXEL_WIDTH  = GRID_WIDTH  * TILE_SIZE   # 600
GRID_PIXEL_HEIGHT = GRID_HEIGHT * TILE_SIZE   # 600
UI_PANEL_X        = GRID_PIXEL_WIDTH          # 600 — where UI panel starts
UI_PANEL_WIDTH    = SCREEN_WIDTH - UI_PANEL_X  # 300

# --- Game ---
MAX_TURNS             = 50
STARTING_ECO_POINTS   = 30

# --- Terrain type constants ---
TERRAIN_LAND      = 'land'
TERRAIN_RIVER     = 'river'
TERRAIN_FARM      = 'farm'
TERRAIN_FOREST    = 'forest'
TERRAIN_RESERVOIR = 'reservoir'
TERRAIN_SOLAR     = 'solar'

# --- Agent identifiers ---
AGENT_A = 'A'
AGENT_B = 'B'

# --- Initial planet meter values (0-100) ---
INITIAL_WATER       = 20.0   # low — below safe zone (50-80), forces agents to act
INITIAL_FOOD        = 20.0   # low — 1.0/step drain means food crisis without farms
INITIAL_OXYGEN      = 20.0   # low — needs forests quickly
INITIAL_TEMPERATURE = 72.0   # slightly high — forces forests to cool planet

# --- Planet meter target ranges ---
METER_OPTIMAL_LOW  = 50.0   # water, food, oxygen
METER_OPTIMAL_HIGH = 80.0
TEMP_OPTIMAL_MIN   = 40.0
TEMP_OPTIMAL_MAX   = 60.0

# --- State abstraction thresholds (for Q-Learning) ---
LOW_THRESHOLD  = 33.3
HIGH_THRESHOLD = 66.6

# --- Simulation rates ---
WATER_FLOW_RATE        = 1.5   # how fast water spreads to neighbors
CROP_GROWTH_RATE       = 2     # turns per growth stage (6 steps to fully mature)
FOREST_MATURATION_TURNS = 3   # turns for a forest to reach maturity (9 steps total)
TEMP_CHANGE_RATE       = 0.5  # degrees changed per farm/forest/solar per turn

# --- Action costs (eco points) ---
ACTION_COSTS = {
    'build_canal':        3,
    'place_reservoir':    6,
    'plant_forest':       4,
    'clear_forest':       2,
    'plant_crop':         3,
    'harvest_crop':       1,
    'build_solar':        8,
    'adjust_allocation':  0,   # free — just sets agent priority mode
}

# --- Q-Learning hyperparameters ---
ALPHA          = 0.1    # learning rate
GAMMA          = 0.9    # discount factor
EPSILON        = 0.2    # exploration rate (epsilon-greedy)
TRAIN_EPISODES = 500    # pre-training episodes before main game

# --- Tile colors (used for placeholder rendering before sprites) ---
TILE_COLORS = {
    TERRAIN_LAND:      (180, 150, 100),
    TERRAIN_RIVER:     ( 64, 164, 223),
    TERRAIN_FARM:      (160, 210,  80),
    TERRAIN_FOREST:    ( 34, 120,  34),
    TERRAIN_RESERVOIR: (  0, 100, 180),
    TERRAIN_SOLAR:     (255, 200,   0),
}

# --- UI colors ---
UI_BG_COLOR      = ( 20,  25,  35)
UI_TEXT_COLOR    = (220, 220, 220)
UI_BORDER_COLOR  = ( 60,  70,  90)
GRID_LINE_COLOR  = ( 40,  40,  40)

AGENT_A_COLOR    = (220,  80,  80)   # red tones
AGENT_B_COLOR    = ( 80, 130, 220)   # blue tones

# --- Meter bar colors ---
METER_COLORS = {
    'water':  ( 64, 164, 223),
    'food':   (160, 210,  80),
    'oxygen': (100, 220, 180),
    'temp':   (230, 130,  50),
}

# --- Stability thresholds ---
STABILITY_HIGH     = 0.8
STABILITY_MODERATE = 0.5
STABILITY_LOW      = 0.2
STABILITY_COLLAPSE = 0.15

# --- Initial river tile positions on 10x10 grid ---
# L-shaped river through the center, equidistant from both agent spawns
INITIAL_RIVER_TILES = [
    (3, 4),
    (4, 4),
    (5, 4),
    (5, 5),
    (6, 5),
]

# --- Agent spawn positions ---
AGENT_A_SPAWN = (1, 1)
AGENT_B_SPAWN = (8, 8)
