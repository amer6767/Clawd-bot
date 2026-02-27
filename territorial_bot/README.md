# Territorial.io AI Bot

A complete AI bot for [Territorial.io](https://territorial.io) with three modular systems:
- **Vision System** – real-time screen capture and territory classification (CNN)
- **Brain System** – strategic decision making (Deep Q-Network / DQN)
- **Action System** – browser automation via Playwright

Models are trained on **Kaggle's free GPU/TPU** and exported for local use.

---

## Project Structure

```
territorial_bot/
├── config.py                    # All tunable settings (aggression, speed, strategy)
├── vision_system.py             # Screen capture + CNN territory classifier
├── brain_system.py              # DQN decision-making AI
├── action_system.py             # Playwright browser controller
├── bot_controller.py            # Main orchestrator + start/stop CLI
├── requirements.txt             # Python dependencies
├── models/                      # Trained model weights (download from Kaggle)
│   ├── vision_model.pth         # CNN vision model
│   ├── brain_model.pth          # DQN brain model
│   └── label_encoder.pkl        # Class label encoder
├── kaggle_notebooks/
│   ├── kaggle_vision_training.ipynb   # Train vision CNN on Kaggle GPU
│   └── kaggle_brain_training.ipynb    # Train DQN brain on Kaggle GPU
├── screenshots/                 # Debug screenshots (auto-created)
└── logs/                        # Bot logs (auto-created)
```

---

## Quick Start

### Step 1 – Install Dependencies

```bash
cd territorial_bot
pip install -r requirements.txt
playwright install chromium
```

### Step 2 – Train Models on Kaggle (Free GPU)

#### Vision Model
1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) → **New Notebook**
2. Upload `kaggle_notebooks/kaggle_vision_training.ipynb`
3. Enable GPU: **Settings → Accelerator → GPU T4 x2**
4. Click **Run All**
5. Download `vision_model.pth` and `label_encoder.pkl` from the **Output** tab
6. Place both files in `territorial_bot/models/`

#### Brain Model
1. Create another Kaggle notebook
2. Upload `kaggle_notebooks/kaggle_brain_training.ipynb`
3. Enable GPU: **Settings → Accelerator → GPU T4 x2**
4. Click **Run All**
5. Download `brain_model.pth` from the **Output** tab
6. Place it in `territorial_bot/models/`

### Step 3 – Run the Bot

```bash
# Basic run (visible browser window)
python bot_controller.py

# Run with debug overlay (shows vision grid)
python bot_controller.py --debug

# Run headless (no browser window)
python bot_controller.py --headless

# Enable online RL training during gameplay
python bot_controller.py --train

# Set custom player name
python bot_controller.py --name "MyAIBot"

# Full options
python bot_controller.py --train --debug --name "AIBot" --log-level DEBUG
```

---

## System Details

### 1. Vision System (`vision_system.py`)

Processes browser screenshots to understand the game state:

| Feature | Description |
|---------|-------------|
| **Screen Capture** | Playwright screenshot → numpy BGR array |
| **Canvas Detection** | Auto-locates game canvas in browser window |
| **Grid Analysis** | Divides canvas into 10×10 grid cells |
| **Cell Classification** | CNN + HSV color heuristics → own/enemy/neutral/border |
| **Border Detection** | Finds cells on territory boundaries |
| **Attack Targeting** | Selects optimal attack target cell |
| **Game Phase** | Detects early/mid/late game |

**CNN Architecture:**
- Input: 64×64 RGB patch
- 3 convolutional blocks (32→64→128 channels)
- Batch normalization + ReLU + MaxPool
- 2 fully-connected layers
- Output: 4-class softmax (own, enemy, neutral, border)

### 2. Brain System (`brain_system.py`)

Makes strategic decisions using a trained DQN:

| Feature | Description |
|---------|-------------|
| **State Encoding** | 64-feature vector from GameState |
| **Action Space** | 9 actions: STAY + 8 directional attacks |
| **DQN Architecture** | 4-layer MLP (256→256→128→9) |
| **Double DQN** | Reduces overestimation bias |
| **Epsilon-Greedy** | Exploration during online training |
| **Reward Shaping** | Territory gain + enemy reduction + time penalty |
| **Rule-Based Fallback** | Heuristic strategy when no model loaded |

**Strategy by game phase:**
- **Early game** (< 5% map): Aggressive expansion into neutral territory
- **Mid game** (5–25%): Balance attack and defense based on aggression setting
- **Late game** (> 25%): Targeted attacks on weakest enemy borders

### 3. Action System (`action_system.py`)

Controls the browser using Playwright:

| Feature | Description |
|---------|-------------|
| **Browser Launch** | Chromium/Firefox/WebKit with anti-detection |
| **Navigation** | Auto-loads `https://territorial.io` |
| **Canvas Detection** | Finds game canvas element |
| **Screenshot** | Captures PNG → numpy array |
| **Attack Move** | Smooth mouse drag from own territory to target |
| **Game Join** | Fills name input + clicks play button |
| **Game Over** | Detects death/game-end, auto-restarts |
| **Troop Slider** | Sets troop allocation ratio |

---

## Configuration (`config.py`)

All bot behavior is tunable via `config.py`:

```python
# Aggression level (0.0 = passive, 1.0 = very aggressive)
BRAIN_CONFIG["aggression"] = 0.65

# Expansion drive (0.0 = consolidate, 1.0 = always expand)
BRAIN_CONFIG["expansion_drive"] = 0.7

# Action speed (ms between actions)
ACTION_CONFIG["action_cooldown_ms"] = 150

# Bot tick rate (ms between decision cycles)
CONTROLLER_CONFIG["tick_rate_ms"] = 500

# Run headless
BROWSER_CONFIG["headless"] = False

# Auto-restart on game over
CONTROLLER_CONFIG["auto_restart"] = True
```

---

## Running Without Trained Models

The bot works immediately without trained models using **rule-based heuristics**:
- Vision: HSV color detection (no CNN needed)
- Brain: Directional heuristics based on territory analysis

Train on Kaggle to significantly improve performance.

---

## Online Training During Gameplay

Run with `--train` to enable online reinforcement learning:

```bash
python bot_controller.py --train
```

The bot will:
1. Play the game using the current model (or heuristics)
2. Store each (state, action, reward, next_state) transition
3. Train the DQN on mini-batches every tick
4. Save model checkpoints every 10 episodes
5. Gradually reduce exploration (epsilon decay)

---

## Troubleshooting

**Playwright not found:**
```bash
pip install playwright
playwright install chromium
```

**PyTorch not found:**
```bash
pip install torch torchvision
```

**Model files missing:**
- The bot will use color heuristics automatically
- Train on Kaggle and download the `.pth` files

**Game canvas not detected:**
- The bot falls back to full viewport
- Adjust `ACTION_CONFIG["canvas_selector"]` if needed

**Bot not joining game:**
- The game UI may have changed; check browser console
- Try running with `--debug` to see what's happening

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    BotController                         │
│  (bot_controller.py)                                     │
│                                                          │
│  ┌──────────────┐  screenshot  ┌──────────────────────┐ │
│  │ ActionSystem │ ──────────→  │   VisionSystem       │ │
│  │              │              │  (CNN + OpenCV)       │ │
│  │  Playwright  │              │  → GameState          │ │
│  │  Chromium    │              └──────────┬───────────┘ │
│  │              │                         │ GameState    │
│  │  territorial │              ┌──────────▼───────────┐ │
│  │  .io         │              │   BrainSystem        │ │
│  │              │  Decision    │  (DQN + Heuristics)  │ │
│  │              │ ←──────────  │  → Decision          │ │
│  └──────────────┘              └──────────────────────┘ │
└─────────────────────────────────────────────────────────┘

Training Pipeline (Kaggle):
  kaggle_vision_training.ipynb  →  vision_model.pth
  kaggle_brain_training.ipynb   →  brain_model.pth
```

---

## License

MIT License – use freely for personal and educational purposes.
