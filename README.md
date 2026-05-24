# EcoForge: World Balance Architects

EcoForge is a Pygame-based AI ecosystem strategy simulation. Two AI agents compete on the same planet by building forests, canals, reservoirs, farms, and solar plants while trying to keep global water, food, oxygen, temperature, population, and stability in balance.

The project compares multiple decision-making approaches in the same environment:

- Minimax with alpha-beta pruning
- Monte Carlo rollouts
- Tabular Q-learning
- Deep Q-Network reinforcement learning

## Project Structure

```text
world_balance_architects/
  main.py                 # Pygame entry point, menus, game loop
  train_dqn.py            # Headless DQN training script
  config.py               # Global constants and tuning values
  requirements.txt        # Basic runtime dependencies
  agents/                 # Minimax, Monte Carlo, Q-learning, DQN agents
  engine/                 # World state, actions, and simulation rules
  render/                 # Pygame rendering and animations
  assets/                 # Sprites, backgrounds, fonts, and UI images
  q_table_A.json          # Saved Q-learning table for Agent A
  q_table_B.json          # Saved Q-learning table for Agent B
  dqn_model_A.pt          # Saved DQN model checkpoint
```

## Features

- Interactive agent selection for Agent A and Agent B.
- Configurable starting planet parameters before each match.
- Automatic AI-vs-AI gameplay with pause, step, reset, and speed controls.
- Ecosystem simulation with water spread, crop growth, forest maturation, population pressure, and cross-resource feedback.
- Shared environmental objective with competitive scoring from owned assets and stable planet management.
- Persistent Q-learning tables and DQN model checkpoints.

## Requirements

- Python 3.10 or newer
- `pygame-ce`
- `numpy`
- `torch` for the DQN agent and DQN training script

Install the basic dependencies from the project folder:

```bash
cd world_balance_architects
python -m pip install -r requirements.txt
```

If you want to use the DQN agent, install PyTorch as well:

```bash
python -m pip install torch
```

For GPU-specific PyTorch builds, use the install command from the official PyTorch selector for your system.

## Running the Game

From the repository root:

```bash
cd world_balance_architects
python main.py
```

The game starts with an agent selection screen. Choose an AI type for Agent A and Agent B, adjust the initial planet settings, then start the match.

## Controls

| Key | Action |
| --- | --- |
| `Space` | Pause or resume auto-play |
| `N` | Advance one turn while paused |
| `R` | Reset and return to agent selection |
| `+` | Increase auto-play speed |
| `-` | Decrease auto-play speed |
| `Esc` | Quit |

## AI Agents

### Minimax

Searches ahead through possible turns using adversarial planning. Alpha-beta pruning and candidate move limits keep the search practical.

### Monte Carlo

Evaluates candidate actions by running random rollouts from each possible move and selecting the action with the best average simulated result.

### Q-Learning

Uses a table of state-action values learned from repeated simulated games. The saved Q-tables are loaded automatically when available.

### DQN

Uses a neural network to estimate action values from continuous world features. The game loads `dqn_model_A.pt` automatically when the DQN agent is selected.

## Training the DQN Agent

Train a DQN model without opening the Pygame window:

```bash
cd world_balance_architects
python train_dqn.py
```

Useful options:

```bash
python train_dqn.py --agent A --episodes 5000
python train_dqn.py --agent B --episodes 8000
```

Training saves a model checkpoint next to the script, such as `dqn_model_A.pt`.

## Gameplay Model

Each agent spends eco points to perform actions on the grid:

- Plant forests to increase oxygen and cool the planet.
- Build canals and reservoirs to spread water.
- Plant and harvest farms to manage food.
- Build solar plants to increase eco-point income.
- Clear forests or adjust allocation priorities when strategically useful.

After each action, the simulation updates water flow, crop growth, forest maturity, global meters, population, stability, score, and eco-point income. The game ends when the ecosystem collapses for several consecutive turns or when the maximum turn count is reached.

## Notes

- The project includes generated/runtime files such as saved Q-tables, Python bytecode, and DQN checkpoints.
- `requirements.txt` currently lists the base dependencies. PyTorch is required only when using DQN features.
- Some source comments and presentation files may show encoding artifacts if opened with the wrong text encoding.
