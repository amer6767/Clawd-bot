"""
Configuration file for Territorial.io AI Bot
Tune aggression, speed, strategy, and all system parameters here.
"""

# ─────────────────────────────────────────────
# GAME URL
# ─────────────────────────────────────────────
GAME_URL = "https://territorial.io"

# ─────────────────────────────────────────────
# BROWSER SETTINGS
# ─────────────────────────────────────────────
BROWSER_CONFIG = {
    "headless": False,           # Set True to run without visible browser
    "window_width": 1280,
    "window_height": 800,
    "browser_type": "chromium",  # "chromium", "firefox", or "webkit"
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# ─────────────────────────────────────────────
# VISION SYSTEM SETTINGS
# ─────────────────────────────────────────────
VISION_CONFIG = {
    # Screenshot capture
    "capture_interval_ms": 200,      # How often to capture screen (ms)
    "frame_width": 640,              # Resize captured frame to this width
    "frame_height": 480,             # Resize captured frame to this height

    # Color detection thresholds (HSV ranges)
    # Own territory color (default blue-ish; adjust after observing game)
    "own_color_hsv_lower": [100, 80, 80],
    "own_color_hsv_upper": [130, 255, 255],

    # Enemy territory (red-ish)
    "enemy_color_hsv_lower": [0, 80, 80],
    "enemy_color_hsv_upper": [15, 255, 255],

    # Neutral/unclaimed territory (gray)
    "neutral_color_hsv_lower": [0, 0, 150],
    "neutral_color_hsv_upper": [180, 30, 220],

    # Model paths
    "vision_model_path": "models/vision_model.pth",
    "label_encoder_path": "models/label_encoder.pkl",

    # Detection confidence threshold
    "confidence_threshold": 0.6,

    # Grid division for territory analysis
    "grid_rows": 10,
    "grid_cols": 10,
}

# ─────────────────────────────────────────────
# BRAIN / DECISION SYSTEM SETTINGS
# ─────────────────────────────────────────────
BRAIN_CONFIG = {
    # Strategy aggression level: 0.0 (passive) to 1.0 (very aggressive)
    "aggression": 0.65,

    # Expansion preference: 0.0 (consolidate) to 1.0 (always expand)
    "expansion_drive": 0.7,

    # Defense threshold: attack own border if enemy pressure > this
    "defense_threshold": 0.4,

    # Game phase thresholds (based on territory % owned)
    "early_game_threshold": 0.05,   # < 5% map owned = early game
    "mid_game_threshold": 0.25,     # 5-25% = mid game
    # > 25% = late game

    # Decision model path
    "brain_model_path": "models/brain_model.pth",

    # State vector size (must match training)
    "state_size": 64,

    # Action space size
    "action_size": 9,  # 8 directions + stay

    # Reinforcement learning params
    "gamma": 0.95,           # Discount factor
    "epsilon_start": 1.0,    # Initial exploration rate
    "epsilon_end": 0.05,     # Minimum exploration rate
    "epsilon_decay": 0.995,  # Decay per episode
    "learning_rate": 0.001,
    "batch_size": 64,
    "memory_size": 10000,
    "target_update_freq": 10,  # Update target network every N episodes
}

# ─────────────────────────────────────────────
# ACTION SYSTEM SETTINGS
# ─────────────────────────────────────────────
ACTION_CONFIG = {
    # Mouse movement speed (pixels per step for smooth movement)
    "mouse_move_steps": 5,
    "mouse_move_delay_ms": 10,

    # Click timing
    "click_hold_ms": 50,       # How long to hold mouse button
    "action_cooldown_ms": 150, # Minimum time between actions

    # Attack drag distance (pixels)
    "attack_drag_distance": 80,

    # Troop allocation slider position (0.0 to 1.0)
    "default_troop_ratio": 0.6,  # Send 60% of troops by default

    # Game canvas selector (CSS selector for the game canvas)
    "canvas_selector": "canvas",

    # Keyboard shortcuts
    "pause_key": "Escape",
}

# ─────────────────────────────────────────────
# BOT CONTROLLER SETTINGS
# ─────────────────────────────────────────────
CONTROLLER_CONFIG = {
    # Main loop tick rate (ms between decision cycles)
    "tick_rate_ms": 500,

    # Screenshot save interval (save every N ticks for debugging)
    "screenshot_save_interval": 50,

    # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
    "log_level": "INFO",

    # Log file path
    "log_file": "logs/bot.log",

    # Max runtime in seconds (0 = unlimited)
    "max_runtime_seconds": 0,

    # Auto-restart on game over
    "auto_restart": True,

    # Restart delay (seconds)
    "restart_delay_seconds": 3,
}

# ─────────────────────────────────────────────
# TRAINING SETTINGS (used by Kaggle notebooks)
# ─────────────────────────────────────────────
TRAINING_CONFIG = {
    # Vision model training
    "vision_epochs": 30,
    "vision_batch_size": 32,
    "vision_learning_rate": 0.001,
    "vision_train_split": 0.8,
    "vision_image_size": (224, 224),
    "vision_num_classes": 4,  # own, enemy, neutral, border

    # Brain model training (DQN)
    "brain_episodes": 1000,
    "brain_max_steps_per_episode": 200,
    "brain_save_interval": 100,  # Save checkpoint every N episodes

    # Data augmentation for vision
    "augment_flip": True,
    "augment_rotate": True,
    "augment_brightness": True,
    "augment_noise": True,

    # Kaggle output directory
    "kaggle_output_dir": "/kaggle/working/",
}

# ─────────────────────────────────────────────
# COLOR PALETTE (for territory identification)
# ─────────────────────────────────────────────
# Known player colors in Territorial.io (RGB)
PLAYER_COLORS = {
    "blue":   (0,   100, 255),
    "red":    (255, 50,  50),
    "green":  (50,  200, 50),
    "yellow": (255, 220, 0),
    "purple": (180, 0,   255),
    "orange": (255, 140, 0),
    "cyan":   (0,   220, 220),
    "pink":   (255, 100, 180),
    "white":  (240, 240, 240),
    "gray":   (150, 150, 150),
}

# Color tolerance for matching (Euclidean distance in RGB space)
COLOR_MATCH_TOLERANCE = 40
