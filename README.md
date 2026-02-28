# Territorial.io AI Bot — Kaggle + ngrok + GitHub Pages

An advanced AI bot for [Territorial.io](https://territorial.io) that runs entirely on **Kaggle's free GPU notebooks** and is controlled remotely from a **mobile-friendly website** hosted on GitHub Pages.

No paid cloud hosting required. No local server needed.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  📱 Mobile Website (GitHub Pages)                           │
│  docs/index.html                                            │
│  → Enter ngrok URL → Start/Stop bot → View screenshot/stats │
└────────────────────┬────────────────────────────────────────┘
                     │  HTTPS (ngrok tunnel)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  🖥️  Kaggle Notebook (Free GPU)                             │
│  territorial_bot_runner.ipynb                               │
│  → Flask API (port 5000) → pyngrok tunnel                   │
│  → BotController → Playwright → territorial.io              │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Open the Kaggle notebook → run all cells → get a public ngrok URL
2. Paste the ngrok URL into the GitHub Pages mobile site
3. Tap **START BOT** — the bot launches headless Chromium on Kaggle and plays
4. Watch the live screenshot and stats update in real time on your phone

---

## How to Run on Kaggle

### Step 1 — Upload bot files to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) → **Create → New Dataset**
2. Upload the entire `territorial_bot/` folder (all `.py` files)
3. Name the dataset `territorial-bot-files`

### Step 2 — Create the runner notebook

1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) → **New Notebook**
2. Upload `territorial_bot/kaggle_notebooks/territorial_bot_runner.ipynb`
3. In **Settings → Accelerator**, select **GPU T4** (optional but faster)
4. In **Settings → Internet**, make sure **Internet is ON** (required for ngrok)
5. Add your dataset: **Add Data → Your Datasets → territorial-bot-files**
   - This mounts your bot files at `/kaggle/input/territorial-bot-files/`
6. In Cell 2, copy the bot files to `/kaggle/working/territorial_bot/`:
   ```python
   import shutil
   shutil.copytree('/kaggle/input/territorial-bot-files/territorial_bot',
                   '/kaggle/working/territorial_bot', dirs_exist_ok=True)
   ```

### Step 3 — Get a free ngrok token

1. Sign up at [ngrok.com](https://ngrok.com) (free)
2. Go to **Your Authtoken** in the dashboard
3. Copy your token

### Step 4 — Configure and run

1. In Cell 5 of the notebook, replace `YOUR_NGROK_AUTH_TOKEN_HERE` with your token:
   ```python
   NGROK_AUTH_TOKEN = "your_actual_token_here"
   ```
2. Click **Run All** (▶▶)
3. Wait for Cell 5 to print your ngrok URL, e.g.:
   ```
   🚀 Bot API is LIVE!
   📱 Your ngrok URL: https://abc123.ngrok.io
   ```
4. Copy that URL

### Step 5 — Control from your phone

1. Open the GitHub Pages site: `https://YOUR_USERNAME.github.io/YOUR_REPO/`
2. Paste the ngrok URL into the input field → tap **Save**
3. Tap **▶ START BOT**
4. Watch the live screenshot and stats update automatically

---

## GitHub Pages Setup

To host the mobile control website:

1. Push this repository to GitHub
2. Go to **Settings → Pages**
3. Set **Source** to `Deploy from a branch`
4. Set **Branch** to `main` and **Folder** to `/docs`
5. Click **Save**
6. Your site will be live at `https://YOUR_USERNAME.github.io/YOUR_REPO/`

---

## Mobile Website Features

The control panel at `docs/index.html` provides:

| Feature | Description |
|---------|-------------|
| **ngrok URL input** | Paste your Kaggle tunnel URL, saved to localStorage |
| **START / STOP buttons** | Large touch-friendly buttons to control the bot |
| **Status display** | Live running/stopped/error state with color coding |
| **Live screenshot** | Auto-refreshes every 2 seconds from `/screenshot` |
| **Stats panel** | Wins, Losses, Territory %, Score — refreshes every 3s |
| **Disconnected state** | Shows "Disconnected" in gray if API is unreachable |

No external dependencies — pure HTML/CSS/JS, works offline after first load.

---

## Bot Systems

| System | File | Description |
|--------|------|-------------|
| **VisionSystem** | `vision_system.py` | CNN + OpenCV territory detection |
| **BrainSystem** | `brain_system.py` | DQN reinforcement learning decisions |
| **ActionSystem** | `action_system.py` | Playwright browser automation |
| **BotController** | `bot_controller.py` | Main orchestrator |
| **EnemyTracker** | `enemy_tracker.py` | Per-enemy behavior profiling |
| **SmartStrategy** | `smart_strategy.py` | Defense, corner, and alliance strategies |
| **LearningSystem** | `learning_system.py` | Win/loss tracking and adaptive parameters |
| **PerformanceOptimizer** | `performance_optimizer.py` | Adaptive quality and frame skipping |
| **AntiDetection** | `anti_detection.py` | Human-like mouse movement and timing |
| **MapAnalyzer** | `map_analyzer.py` | Cluster detection, chokepoints, expansion paths |
| **StatsDashboard** | `stats_dashboard.py` | Real-time metrics and session reports |

---

## Kaggle Notebooks

| Notebook | Purpose |
|----------|---------|
| `territorial_bot_runner.ipynb` | **Run the bot** with Flask API + ngrok tunnel |
| `kaggle_brain_training.ipynb` | Train the DQN brain model on Kaggle GPU |
| `kaggle_vision_training.ipynb` | Train the CNN vision model on Kaggle GPU |

### Training Models (Optional)

The bot works immediately with rule-based heuristics. To train AI models:

1. Run `kaggle_vision_training.ipynb` on Kaggle GPU → download `vision_model.pth`
2. Run `kaggle_brain_training.ipynb` on Kaggle GPU → download `brain_model.pth`
3. Upload both `.pth` files to your Kaggle dataset under `territorial_bot/models/`

---

## Project Structure

```
territorial_bot/
├── config.py                    # All tunable settings
├── vision_system.py             # Screen capture + CNN territory classifier
├── brain_system.py              # DQN decision-making AI
├── action_system.py             # Playwright browser controller
├── bot_controller.py            # Main orchestrator (integrates all systems)
├── enemy_tracker.py             # Individual enemy tracking & behavior analysis
├── smart_strategy.py            # Advanced strategic decision making
├── learning_system.py           # Performance tracking & adaptive learning
├── performance_optimizer.py     # CPU/memory optimization & adaptive quality
├── anti_detection.py            # Human-like behavior simulation
├── map_analyzer.py              # Advanced spatial analysis & expansion planning
├── stats_dashboard.py           # Real-time statistics & reporting
├── requirements.txt             # Python dependencies (includes flask, pyngrok)
├── models/                      # Trained model weights (.pth files)
├── kaggle_notebooks/
│   ├── territorial_bot_runner.ipynb   # ← Run this on Kaggle to play
│   ├── kaggle_brain_training.ipynb    # Train DQN brain model
│   ├── kaggle_vision_training.ipynb   # Train CNN vision model
│   ├── PASTE_THIS_brain_training.py   # Standalone training script
│   └── PASTE_THIS_vision_training.py  # Standalone training script
├── screenshots/                 # Debug screenshots (auto-created)
└── logs/                        # Bot logs & session reports

docs/
└── index.html                   # Mobile control website (GitHub Pages)
```

---

## Requirements

```
playwright>=1.40.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
tqdm>=4.65.0
flask>=3.0.0
pyngrok>=7.0.0
```

---

## License

MIT License — use freely for personal and educational purposes.
