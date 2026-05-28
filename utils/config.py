"""Application-wide configuration for Spite Analysis."""

# ── Window ─────────────────────────────────────────────────────────────────
WINDOW_TITLE = "Spite Analysis"
WINDOW_MIN_W = 1100
WINDOW_MIN_H = 750

# ── Card dimensions ────────────────────────────────────────────────────────
CARD_W = 72
CARD_H = 100
CARD_RADIUS = 8          # rounded corner radius
CARD_SPACING = 8         # gap between hand cards

# ── Colours ────────────────────────────────────────────────────────────────
COLOR_TABLE     = "#2d5a27"   # dark green felt
COLOR_TABLE_ALT = "#245020"
COLOR_CARD_FACE = "#fefefe"
COLOR_CARD_BACK = "#1a3c8a"
COLOR_RED_CARD  = "#cc1111"
COLOR_BLACK_CARD = "#111111"

COLOR_HIGHLIGHT    = "#ffe066"   # selected card
COLOR_LEGAL_GLOW   = "#66ff99"   # legal target
COLOR_STOCKPILE_BG = "#1a2a5a"
COLOR_DISCARD_BG   = "#3a2a1a"
COLOR_BUILD_BG     = "#1a3a1a"
COLOR_EMPTY_PILE   = "#1a4a1a"

COLOR_WILD_RED_GLOW   = "#ff4444"
COLOR_WILD_BLACK_GLOW = "#444444"

COLOR_UI_BG     = "#1e1e2e"
COLOR_UI_PANEL  = "#2a2a3e"
COLOR_UI_TEXT   = "#e0e0f0"
COLOR_UI_ACCENT = "#7c6af7"
COLOR_BUTTON    = "#4a4a7a"
COLOR_BUTTON_HOVER = "#6a6aaa"

# ── Timing ─────────────────────────────────────────────────────────────────
AI_THINK_DELAY_MS  = 600     # ms between AI actions in normal play
AI_FAST_DELAY_MS   = 50      # ms in fast/training mode
AI_STEP_DELAY_MS   = 0       # ms in headless mode

# ── Training defaults ──────────────────────────────────────────────────────
DEFAULT_TRAIN_EPISODES = 1000
DEFAULT_CHECKPOINT_DIR = "checkpoints"
DEFAULT_STATS_PATH     = "training_stats.json"

# ── AI agents ─────────────────────────────────────────────────────────────
AGENT_TYPES = ["Random", "Heuristic", "RL (DQN)"]
DEFAULT_AI_TYPE = "Heuristic"
