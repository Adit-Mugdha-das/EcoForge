# EcoForge — World Balance Architects | Copilot Working Context

## Project Overview
This is an AI-powered turn-based environmental strategy game built in Python.
Two AI agents (Agent A and Agent B) compete to maintain a planet's ecosystem
across a 25×18 grid over 150 turns. The backend is fully complete. My job is
to build and polish the frontend using Pygame.

---

## Tech Stack
- **Python 3.13**
- **pygame-ce** (community edition — use this, NOT plain pygame)
- **Pillow** (PIL) — for image processing and tile manipulation
- Language: Python only. No web, no JavaScript.

---

## Project Structure
```
world_balance_architects/
├── main.py                  ← entry point, game loop (backend done)
├── config.py                ← all constants (GRID_WIDTH=25, GRID_HEIGHT=18, TILE_SIZE=48, etc.)
├── requirements.txt
├── assets/
│   └── tiles/               ← 48×48 pixel-art PNGs (already downloaded)
│       ├── land.png
│       ├── river_1.png      ← water frame 1
│       ├── river_2.png      ← water frame 2 (animation)
│       ├── forest.png
│       ├── farm.png
│       ├── reservoir.png
│       ├── solar_plant.png
│       ├── farm_stage_1/2/3.png     ← crop growth overlays
│       ├── forest_mat_2/3.png       ← forest maturity overlays
│       ├── owner_a_tint.png         ← Agent A territory overlay
│       ├── owner_b_tint.png         ← Agent B territory overlay
│       ├── highlight.png            ← last-action cell glow
│       └── highlight_green.png
├── engine/
│   ├── world.py             ← World class, Cell class (DONE)
│   ├── simulate.py          ← simulation rules (DONE)
│   └── actions.py           ← all 8 actions (DONE)
├── agents/
│   ├── base_agent.py        ← abstract base (DONE)
│   ├── dqn_agent.py         ← Deep Q-Network agent (DONE)
│   ├── q_learning.py        ← Q-Learning agent (DONE)
│   ├── minimax.py           ← Minimax + alpha-beta (DONE)
│   ├── monte_carlo.py       ← MCTS agent (DONE)
│   └── eval.py              ← evaluation functions (DONE)
└── render/
    ├── renderer.py          ← MY FILE — already downloaded, actively editing
    └── animations.py        ← helper classes (FloatingText, Particle, CameraShake, WaterAnimator)
```

---

## World & Cell Model (backend — DO NOT modify)

```python
class Cell:
    terrain: str          # "land" | "river" | "farm" | "forest" | "reservoir" | "solar_plant"
    owner: str | None     # "A" | "B" | None
    water_value: int      # 0–10
    crop_stage: int       # 0–3  (farm tiles only)
    forest_maturity: int  # 1–3  (forest tiles only)

class World:
    grid: list[list[Cell]]   # grid[row][col]  — ROW-MAJOR
    water_level: float       # 0–100
    food: float              # 0–100
    oxygen: float            # 0–100
    temperature: float       # 0–100  (optimal = 40–60)
    stability: float         # 0.0–1.0
    turn: int
    current_agent: str       # "A" or "B"
    scores: dict             # {"A": float, "B": float}
    eco_points: dict         # {"A": int,   "B": int}
    action_log: list[str]    # last N action descriptions
```

---

## Config Constants (from config.py — already set)

```python
GRID_WIDTH, GRID_HEIGHT = 25, 18
TILE_SIZE               = 48
SCREEN_WIDTH            = 1480    # grid (1200px) + UI panel (280px)
SCREEN_HEIGHT           = 900
MAX_TURNS               = 150
FPS                     = 60

TERRAIN_LAND       = "land"
TERRAIN_RIVER      = "river"
TERRAIN_FARM       = "farm"
TERRAIN_FOREST     = "forest"
TERRAIN_RESERVOIR  = "reservoir"
TERRAIN_SOLAR      = "solar_plant"

AGENT_A, AGENT_B         = "A", "B"
AGENT_A_COLOR            = (220, 80,  60)   # red
AGENT_B_COLOR            = ( 60, 140, 220)  # blue

UI_PANEL_X     = GRID_WIDTH * TILE_SIZE     # = 1200
UI_PANEL_WIDTH = SCREEN_WIDTH - UI_PANEL_X  # = 280

STABILITY_HIGH, STABILITY_MODERATE, STABILITY_LOW = 0.75, 0.50, 0.25
```

---

## Renderer API (renderer.py — my file)

The `Renderer` class must expose this interface so `main.py` can call it:

```python
renderer = Renderer(screen)          # init

# Every frame:
renderer.draw(world)                 # full frame render

# Trigger visual effects from game loop:
renderer.set_last_action(row, col)   # highlight cell after AI move
renderer.set_thinking(agent)         # show pulsing dot while AI computes (None to clear)
renderer.trigger_action_fx(action_name, col, row)  # spawn float + particles
renderer.camera_shake.trigger()      # shake on critical events
renderer.add_float(text, col, row, color)
renderer.add_particles(col, row, color, count)

# Game over:
renderer.draw_game_over(world, winner)
```

**Action names for trigger_action_fx:**
`"harvest_crop"`, `"plant_forest"`, `"plant_crop"`, `"build_canal"`,
`"build_solar"`, `"plant_reservoir"`, `"clear_forest"`

---

## Visual Effects Already Implemented (in renderer.py)

| Feature | Status | Notes |
|---|---|---|
| Pixel-art tile rendering | ✅ Done | Pillow → pygame-ce, NEAREST scale |
| Animated water (2-frame) | ✅ Done | WaterAnimator, 28-tick interval |
| Owner tint overlays | ✅ Done | Semi-transparent colored tile |
| Grid lines | ✅ Done | SRCALPHA line surface |
| Action highlight | ✅ Done | Fades out over 35 frames |
| Day/Night cycle | ✅ Done | 4 phases over 20 turns |
| Scanline overlay | ✅ Done | Retro CRT feel |
| FloatingText | ✅ Done | Rises + fades on actions |
| Particle burst | ✅ Done | On harvest/plant |
| Camera shake | ✅ Done | CameraShake class |
| Meter warning flash | ✅ Done | Pulsing red when < 20 |
| AI thinking dot | ✅ Done | Breathing animated circle |
| Game-over overlay | ✅ Done | Gold banner, scores, prompt |
| Tile legend | ✅ Done | Bottom of UI panel |

---

## Visual Style Rules (ALWAYS follow these)

1. **Pixel art** — all tile images are 48×48. Always scale with `Image.NEAREST` (Pillow) or `pygame.transform.scale` — NEVER use smoothscale on tiles.
2. **Color palette** — dark space background `(12, 14, 22)`, deep navy UI panel `(18, 22, 36)`, blue accent `(80, 180, 255)`. Avoid bright whites on the grid.
3. **Font** — use `Consolas` or `monospace` system font (pixel-game feel). If a `.ttf` pixel font exists at `assets/fonts/pixel.ttf`, use it.
4. **Grid is ROW-MAJOR** — `world.grid[row][col]`, screen x = `col * TILE_SIZE`, screen y = `row * TILE_SIZE`.
5. **UI panel** starts at x=1200, width=280. Never draw game tiles into the panel area.
6. **pygame-ce only** — import as `import pygame`. It's a drop-in replacement.
7. **SRCALPHA surfaces** for all transparency/overlays.
8. **60 FPS cap** — use `clock.tick(60)` in main loop.

---

## What I'm Working On Next (ask Copilot for help with these)

- [ ] Speed control (slow/normal/fast sim speed toggle)
- [ ] Pause / step-by-step mode (spacebar to pause, N to step one turn)
- [ ] Mini-map in the UI panel showing the full grid at small scale
- [ ] Agent algorithm selector UI (choose which AI plays A vs B)
- [ ] Stats graph overlay (turn-by-turn meter history line charts)
- [ ] Tooltip on hover — show cell terrain, owner, water value, crop stage
- [ ] Sound effects using `pygame.mixer` (harvest chime, build thud, critical alarm)
- [ ] Replay system — record all actions, play back any game

---

## How to Run

```bash
pip install pygame-ce Pillow
cd world_balance_architects
python main.py
```

---

## Copilot Instructions

- When I ask for a new feature, always edit `render/renderer.py` or `render/animations.py` unless I say otherwise.
- Never touch `engine/`, `agents/`, or `config.py` — those are complete.
- Always keep the `Renderer` public API intact (draw, set_last_action, set_thinking, trigger_action_fx, draw_game_over).
- When adding a new visual system, add it as a new method prefixed with `_draw_` or `_update_` and call it from `draw()`.
- All new animation helpers (like new particle types or new float styles) go in `render/animations.py`, not inline in renderer.py.
- Prefer pygame-ce drawing primitives (`pygame.draw.*`, `pygame.Surface`, `SRCALPHA`) over loading new image files unless I explicitly ask for new tiles.
- If you need a new tile variant, generate it with Pillow (using pixel-art style matching the existing palette) and save it to `assets/tiles/`.