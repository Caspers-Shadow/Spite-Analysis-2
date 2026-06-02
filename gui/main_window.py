"""
MainWindow – complete application window for Spite Analysis.

Key fixes in this version
--------------------------
* TrainingScreen shows clear ROLE labels (Learner vs Opponent) and a live
  description of what the current agent combination does.
* Chart generation is moved to a background thread (ChartDataWorker) so the
  GUI never freezes while charts are being computed.  A progress bar shows
  "Computing charts…" during generation.
* self._thread.wait() removed from _on_finished (was the main cause of
  "not responding" after training ended).
"""

import json, os
from typing import Optional, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox,
    QMessageBox, QFileDialog, QSlider,
    QGroupBox, QDialog, QDialogButtonBox,
    QFormLayout, QApplication, QProgressBar,
    QFrame,
)
from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QFont

from game.game_state import GameState, Action
from game.player import Player
from ai.agent import Agent
from ai.random_agent import RandomAgent
from ai.heuristic_agent import HeuristicAgent
from ai.rl_agent import RLAgent
from ai.mcts_agent import MCTSAgent
from ai.training import TrainingSession
from gui.board_view import BoardView
from gui.statistics_panel import StatisticsPanel
from gui.stats_view import StatsViewScreen
from gui.rules_screen import RulesScreen
from utils.stats_manager import StatsManager
from utils.config import (
    WINDOW_TITLE, WINDOW_MIN_W, WINDOW_MIN_H,
    AI_THINK_DELAY_MS, AI_FAST_DELAY_MS,
    COLOR_UI_BG, COLOR_UI_TEXT, COLOR_UI_ACCENT,
    COLOR_BUTTON, COLOR_BUTTON_HOVER, DEFAULT_TRAIN_EPISODES,
)
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False

_stats_manager = StatsManager("game_stats.json")

# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────
BTN = f"""
    QPushButton {{
        background:{COLOR_BUTTON}; color:{COLOR_UI_TEXT}; border:none;
        border-radius:8px; padding:10px 22px; font-size:13px; font-weight:bold;
    }}
    QPushButton:hover   {{ background:{COLOR_BUTTON_HOVER}; }}
    QPushButton:pressed {{ background:{COLOR_UI_ACCENT}; }}
    QPushButton:disabled {{ background:#333344; color:#555566; }}
"""
SMALL = BTN.replace("13px","11px").replace("10px 22px","6px 14px").replace("8px","6px")

def _mkbtn(text, color=None, small=False):
    b = QPushButton(text)
    style = SMALL if small else BTN
    b.setStyleSheet(style.replace(COLOR_BUTTON, color, 1) if color else style)
    return b

def _lbl(text, bold=False, size=10, color=COLOR_UI_TEXT):
    l = QLabel(text)
    f = QFont("Arial", size); f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color:{color}; background:transparent;")
    return l

def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{ color:{COLOR_UI_ACCENT}; font-weight:bold; font-size:10px;
            border:1px solid {COLOR_UI_ACCENT}; border-radius:5px;
            margin-top:6px; background:#2a2a3e; }}
        QGroupBox::title {{ subcontrol-origin:margin; padding:0 4px; }}""")
    return g

def _make_agent(t: str, pid: int) -> Agent:
    if t == "Random":         return RandomAgent(pid)
    if t == "RL (DQN)":       return RLAgent(pid)
    if t.startswith("MCTS"):
        n = int(t.split("-")[1]) if "-" in t else 15
        return MCTSAgent(pid, rollouts=n)
    return HeuristicAgent(pid)

def _agent_role(t: str):
    """Return (emoji, role_text, color) for an agent type string."""
    if "DQN" in t:
        return "🎓", "Learner (trains via experience)", "#50c878"
    if t.startswith("MCTS"):
        return "🔍", "Search Agent (thinks, no training)", "#ffe066"
    if t == "Heuristic":
        return "📋", "Rule-based Agent (strategic)", "#aaaaff"
    if t == "Random":
        return "🎲", "Random Baseline (no strategy)", "#ff9999"
    return "🤖", "AI Agent", COLOR_UI_TEXT

AGENT_PLAY_OPTIONS  = ["Heuristic", "MCTS-15", "MCTS-30", "Random", "RL (DQN)"]
AGENT_TRAIN_OPTIONS = ["Heuristic", "Random", "MCTS-5", "MCTS-15", "RL (DQN)"]


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class AIActionWorker(QObject):
    done = pyqtSignal(object)
    def __init__(self, agent, state):
        super().__init__()
        self.agent = agent
        self.state = state
    def run(self):
        try:
            action = self.agent.choose_action(self.state)
        except Exception as e:
            log.error(f"AI worker error: {e}")
            action = None
        self.done.emit(action)


class TrainingWorker(QObject):
    progress = pyqtSignal(int, dict)
    finished = pyqtSignal(dict)
    def __init__(self, session):
        super().__init__()
        self.session = session
        self._running = True
    def run(self):
        def cb(sess):
            if not self._running: sess.stop(); return
            if sess.episode % 10 == 0:
                self.progress.emit(sess.episode, sess.get_summary())
        self.session.run(callback=cb)
        self.finished.emit(self.session.get_summary())
    def stop(self):
        self._running = False
        self.session.stop()


class ChartDataWorker(QObject):
    """
    Computes all chart data arrays in a background thread (pure Python,
    no matplotlib / Qt calls).  Emits done(data_dict) when finished.
    The main thread then draws the figure from the data dict — fast and safe.
    """
    done     = pyqtSignal(dict)
    progress = pyqtSignal(int)    # 0-100 percent

    def __init__(self, results, a0_label, a1_label):
        super().__init__()
        self.results  = results[:]   # safe copy — no shared state
        self.a0 = a0_label
        self.a1 = a1_label

    def run(self):
        results = self.results
        if not results:
            self.done.emit({})
            return

        n      = len(results)
        turns  = [r["turns"] for r in results]
        window = min(50, max(10, n // 10))

        self.progress.emit(10)

        # Rolling win-rate for P0
        xs, ys = [], []
        for i in range(1, n + 1):
            chunk = results[max(0, i - window):i]
            xs.append(i)
            ys.append(sum(1 for r in chunk if r["winner"] == 0) / len(chunk))

        self.progress.emit(40)

        # Smoothed game length
        smooth = [sum(turns[max(0, i-20):i+1]) / min(i+1, 20)
                  for i in range(len(turns))]

        self.progress.emit(70)

        wins = [
            sum(1 for r in results if r["winner"] == 0),
            sum(1 for r in results if r["winner"] == 1),
        ]

        self.progress.emit(100)

        self.done.emit({
            'xs': xs, 'ys': ys, 'window': window,
            'smooth': smooth, 'wins': wins,
            'total': n, 'a0': self.a0, 'a1': self.a1,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Embedded chart widget
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddedChart(QWidget):
    def __init__(self, figsize=(10, 4), parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#1e1e2e;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if _MPL:
            self._fig    = Figure(figsize=figsize, facecolor="#1e1e2e", tight_layout=True)
            self._canvas = FigureCanvas(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(_lbl("matplotlib not installed – pip install matplotlib",
                                  color="#ff7f7f"))
        self._show_waiting()

        # Worker refs kept alive
        self._chart_thread: Optional[QThread] = None
        self._chart_worker: Optional[ChartDataWorker] = None

    # ── State messages ─────────────────────────────────────────────────────────
    def _show_waiting(self):
        if not _MPL: return
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#1e1e2e")
        ax.text(0.5, 0.5, "Run a training session to see results here",
                ha="center", va="center", color="#555566", fontsize=13,
                transform=ax.transAxes)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        self._canvas.draw()

    def _show_computing(self):
        if not _MPL: return
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor("#1e1e2e")
        ax.text(0.5, 0.5, "⏳  Computing chart data…",
                ha="center", va="center", color=COLOR_UI_ACCENT, fontsize=14,
                transform=ax.transAxes)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        self._canvas.draw()

    # ── Async rendering ────────────────────────────────────────────────────────
    def render_async(self, session: TrainingSession, a0: str, a1: str,
                     on_done=None, on_progress=None):
        """
        Compute chart data in a background thread; draw on the main thread.
        on_done()     called when chart is fully drawn.
        on_progress(pct) called with 0-100 during computation.
        """
        self._show_computing()

        results = session.results
        if not results:
            if on_done: on_done()
            return

        self._chart_worker = ChartDataWorker(results, a0, a1)
        self._chart_thread = QThread()
        self._chart_worker.moveToThread(self._chart_thread)
        self._chart_thread.started.connect(self._chart_worker.run)

        if on_progress:
            self._chart_worker.progress.connect(on_progress)

        def _on_data(data):
            self._draw_from_data(data)
            if on_done: on_done()

        self._chart_worker.done.connect(_on_data)
        self._chart_worker.done.connect(self._chart_thread.quit)
        self._chart_thread.finished.connect(self._chart_thread.deleteLater)
        self._chart_thread.start()

    # ── Drawing (main thread only) ─────────────────────────────────────────────
    def _draw_from_data(self, data: dict):
        if not _MPL or not data: return
        BG, PANEL = "#1e1e2e", "#2a2a3e"
        ACC, GRN, SAL, GRY = "#7c6af7", "#50c878", "#ff7f7f", "#888899"

        self._fig.clf()
        axes = self._fig.subplots(1, 3)

        xs, ys     = data['xs'], data['ys']
        smooth     = data['smooth']
        wins       = data['wins']
        window     = data['window']
        a0, a1     = data['a0'], data['a1']
        total      = data['total']

        # ── Panel 1: rolling win rate ──────────────────────────────────────────
        ax = axes[0]
        ax.set_facecolor(PANEL)
        ax.plot(xs, ys, color=ACC, linewidth=2)
        ax.axhline(0.5, color=GRY, linestyle="--", linewidth=1)
        ax.fill_between(xs, ys, 0.5, where=[y>0.5 for y in ys],  alpha=0.2, color=GRN)
        ax.fill_between(xs, ys, 0.5, where=[y<=0.5 for y in ys], alpha=0.2, color=SAL)
        ax.set_ylim(0, 1)
        ax.set_title(f"{a0}  Win Rate (rolling {window})", color=COLOR_UI_TEXT, fontsize=9)
        ax.set_xlabel("Episode", color=GRY, fontsize=8)
        try:
            import matplotlib.ticker as tk
            ax.yaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f"{v:.0%}"))
        except Exception: pass
        for s in ax.spines.values(): s.set_color("#333344")
        ax.tick_params(colors=GRY, labelsize=7)

        # ── Panel 2: smoothed game length ──────────────────────────────────────
        ax2 = axes[1]
        ax2.set_facecolor(PANEL)
        ax2.plot(range(len(smooth)), smooth, color=GRN, linewidth=1.5)
        ax2.set_title("Avg Game Length (rolling 20)", color=COLOR_UI_TEXT, fontsize=9)
        ax2.set_xlabel("Episode", color=GRY, fontsize=8)
        ax2.set_ylabel("Turns", color=GRY, fontsize=8)
        for s in ax2.spines.values(): s.set_color("#333344")
        ax2.tick_params(colors=GRY, labelsize=7)
        ax2.grid(True, alpha=0.15, color=GRY)

        # ── Panel 3: wins bar chart ────────────────────────────────────────────
        ax3 = axes[2]
        ax3.set_facecolor(PANEL)
        bars = ax3.bar([a0, a1], wins, color=[ACC, SAL], alpha=0.85)
        for bar, val in zip(bars, wins):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val}\n({val/total*100:.1f}%)",
                     ha="center", color=COLOR_UI_TEXT, fontsize=8)
        ax3.set_title(f"Results  ({total} games)", color=COLOR_UI_TEXT, fontsize=9)
        ax3.set_ylabel("Wins", color=GRY, fontsize=8)
        for s in ax3.spines.values(): s.set_color("#333344")
        ax3.tick_params(colors=GRY, labelsize=8)
        ax3.grid(True, alpha=0.15, axis="y", color=GRY)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Main Menu
# ─────────────────────────────────────────────────────────────────────────────

class MenuScreen(QWidget):
    human_vs_ai = pyqtSignal()
    ai_vs_ai    = pyqtSignal()
    start_train = pyqtSignal()
    view_stats  = pyqtSignal()
    view_rules  = pyqtSignal()
    exit_app    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(16)

        t = QLabel("♠  SPITE ANALYSIS  ♠")
        t.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        t.setStyleSheet(f"color:{COLOR_UI_ACCENT}; background:transparent; letter-spacing:4px;")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)

        s = QLabel("A card game of skill, strategy & spite")
        s.setFont(QFont("Arial", 12))
        s.setStyleSheet("color:#888899; background:transparent;")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(s)
        lay.addSpacing(20)

        for text, sig, color in [
            ("🧑  Human vs AI",      self.human_vs_ai,  "#2a5a2a"),
            ("🤖  AI vs AI",         self.ai_vs_ai,     "#2a2a5a"),
            ("🎓  Start Training",   self.start_train,  "#5a3a1a"),
            ("📊  View Statistics",  self.view_stats,   "#3a1a5a"),
            ("📖  Rules & Help",     self.view_rules,   "#1a3a5a"),
            ("✖   Exit",            self.exit_app,     "#5a1a1a"),
        ]:
            b = _mkbtn(text, color); b.setFixedWidth(290)
            b.clicked.connect(sig.emit)
            lay.addWidget(b, alignment=Qt.AlignmentFlag.AlignCenter)


# ─────────────────────────────────────────────────────────────────────────────
# Mode config dialog
# ─────────────────────────────────────────────────────────────────────────────

class ModeDialog(QDialog):
    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure: {mode}")
        self.setStyleSheet(f"background:{COLOR_UI_BG}; color:{COLOR_UI_TEXT};")
        self.setMinimumWidth(340)
        lay = QFormLayout(self); lay.setSpacing(10)
        cs = f"background:#333344; color:{COLOR_UI_TEXT}; padding:4px;"
        ls = f"color:{COLOR_UI_TEXT}; font-size:11px;"

        if mode == "Human vs AI":
            self.ai_combo = QComboBox()
            self.ai_combo.addItems(AGENT_PLAY_OPTIONS)
            self.ai_combo.setStyleSheet(cs)
            l = QLabel("AI Opponent:"); l.setStyleSheet(ls)
            lay.addRow(l, self.ai_combo)
            tip = QLabel("MCTS-15 = strongest  |  Heuristic = fastest")
            tip.setStyleSheet("color:#888899; font-size:9px;")
            lay.addRow(tip)
        else:
            self.ai0_combo = QComboBox()
            self.ai0_combo.addItems(AGENT_PLAY_OPTIONS)
            self.ai0_combo.setStyleSheet(cs)
            self.ai1_combo = QComboBox()
            self.ai1_combo.addItems(AGENT_PLAY_OPTIONS)
            self.ai1_combo.setCurrentText("Random")
            self.ai1_combo.setStyleSheet(cs)
            for txt, w in [("Player 1:", self.ai0_combo), ("Player 2:", self.ai1_combo)]:
                l = QLabel(txt); l.setStyleSheet(ls)
                lay.addRow(l, w)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.setStyleSheet(f"color:{COLOR_UI_TEXT};")
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addRow(bb)


# ─────────────────────────────────────────────────────────────────────────────
# Game Screen
# ─────────────────────────────────────────────────────────────────────────────

class GameScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: Optional[GameState] = None
        self.agents: List[Optional[Agent]] = [None, None]
        self.human_idx: int = 0
        self.ai_delay:  int = AI_THINK_DELAY_MS
        self._p0_label = "Human"
        self._p1_label = "AI"
        self._games_played = 0
        self._wins: List[int] = [0, 0]
        self._total_turns = 0
        self._ai_timer = QTimer()
        self._ai_timer.setSingleShot(True)
        self._ai_timer.timeout.connect(self._start_ai_turn)
        self._ai_thread: Optional[QThread] = None
        self._ai_worker: Optional[AIActionWorker] = None
        self._ai_busy = False
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        self._board = BoardView()
        self._board.action_requested.connect(self._on_human_action)

        tb = QWidget(); tb.setStyleSheet("background:rgba(10,10,20,200);")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(10,4,10,4)
        self._btn_menu = _mkbtn("← Menu","#333355",small=True); self._btn_menu.setFixedWidth(100)
        self._btn_menu.clicked.connect(self._confirm_menu)
        self._btn_new  = _mkbtn("New Game","#225522",small=True); self._btn_new.setFixedWidth(120)
        self._btn_new.clicked.connect(self._start_new_game)
        tbl.addWidget(self._btn_menu); tbl.addWidget(self._btn_new); tbl.addStretch()
        self._mode_lbl = QLabel("")
        self._mode_lbl.setStyleSheet(f"color:{COLOR_UI_ACCENT}; font-weight:bold; font-size:11px;")
        tbl.addWidget(self._mode_lbl); tbl.addStretch()
        sl = QLabel("AI Speed:"); sl.setStyleSheet(f"color:{COLOR_UI_TEXT}; font-size:10px;")
        tbl.addWidget(sl)
        self._speed = QSlider(Qt.Orientation.Horizontal)
        self._speed.setRange(0,4); self._speed.setValue(2); self._speed.setFixedWidth(100)
        self._speed.valueChanged.connect(
            lambda v: setattr(self,'ai_delay',[2000,1000,600,200,50][v]))
        tbl.addWidget(self._speed)

        bw = QWidget(); bwl = QVBoxLayout(bw)
        bwl.setContentsMargins(0,0,0,0); bwl.setSpacing(0)
        bwl.addWidget(tb); bwl.addWidget(self._board, stretch=1)
        root.addWidget(bw, stretch=1)

        self._stats = StatisticsPanel()
        root.addWidget(self._stats)

    def start_human_vs_ai(self, ai_type="Heuristic"):
        self.human_idx = 0
        self.agents = [None, _make_agent(ai_type, 1)]
        self._p0_label = "Human"; self._p1_label = ai_type
        self._mode_lbl.setText(f"Human vs {ai_type}")
        self._games_played = 0; self._wins = [0,0]; self._total_turns = 0
        self._start_new_game()

    def start_ai_vs_ai(self, a0="Heuristic", a1="Random", delay=AI_FAST_DELAY_MS):
        self.human_idx = -1
        self.agents = [_make_agent(a0,0), _make_agent(a1,1)]
        self._p0_label = a0; self._p1_label = a1
        self.ai_delay = delay
        self._mode_lbl.setText(f"{a0} vs {a1}")
        self._games_played = 0; self._wins = [0,0]; self._total_turns = 0
        self._start_new_game()

    def _start_new_game(self):
        self._stop_ai_worker(); self._ai_timer.stop(); self._ai_busy = False
        p0n = "Human" if self.human_idx == 0 else self._p0_label
        p1n = "Human" if self.human_idx == 1 else self._p1_label
        self.state = GameState(player_names=(p0n, p1n))
        disp = max(self.human_idx, 0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        self._board.set_interactive(self.human_idx >= 0 and
                                    self.state.current_player_idx == self.human_idx)
        self._board.set_status(
            "Select a card to play, or click End Turn to discard."
            if self.human_idx >= 0 else "AI vs AI – watching…")
        self._maybe_schedule_ai()

    def _on_human_action(self, action: Action):
        if not self.state or self.state.game_over or self._ai_busy: return
        if self.state.current_player_idx != self.human_idx: return
        ok,_ = self.state.apply_action(action)
        if not ok:
            self._board.set_status("❌ Invalid move – try again"); return
        disp = max(self.human_idx,0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        if self.state.game_over:
            self._handle_game_over(); return
        if self.state.current_player_idx != self.human_idx:
            self._board.set_interactive(False)
            self._board.set_status("AI is thinking…")
            self._ai_timer.start(self.ai_delay)
        else:
            refill = ("  ✨ Fresh hand drawn – keep playing!"
                      if action.action_type.startswith("play") and
                      len(self.state.current_player.hand) == Player.MAX_HAND_SIZE
                      and self.state.hand_refills > 0 else "")
            self._board.set_status("Play a card or click End Turn to discard." + refill)

    def _maybe_schedule_ai(self):
        if not self.state or self.state.game_over or self._ai_busy: return
        if self.agents[self.state.current_player_idx] is not None:
            self._ai_timer.start(self.ai_delay)

    def _start_ai_turn(self):
        if not self.state or self.state.game_over or self._ai_busy: return
        pid   = self.state.current_player_idx
        agent = self.agents[pid]
        if agent is None: return
        self._ai_busy = True
        self._ai_worker = AIActionWorker(agent, self.state.copy())
        self._ai_thread = QThread()
        self._ai_worker.moveToThread(self._ai_thread)
        self._ai_thread.started.connect(self._ai_worker.run)
        self._ai_worker.done.connect(self._on_ai_action_computed)
        self._ai_worker.done.connect(self._ai_thread.quit)
        self._ai_thread.finished.connect(self._ai_thread.deleteLater)
        self._ai_thread.start()

    def _on_ai_action_computed(self, action):
        self._ai_busy = False
        if not self.state or self.state.game_over: return
        if action is None:
            discards = self.state.get_discard_actions()
            action = discards[0] if discards else None
        if action is None: return
        pid   = self.state.current_player_idx
        agent = self.agents[pid]
        self.state.apply_action(action)
        disp = max(self.human_idx,0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        if agent: self._stats.update_ai_info(agent, str(action))
        if self.state.game_over:
            self._handle_game_over(); return
        new_pid = self.state.current_player_idx
        if self.agents[new_pid] is not None:
            self._ai_timer.start(self.ai_delay)
        else:
            self._board.set_interactive(True)
            self._board.set_status("Your turn! Play a card or click End Turn.")

    def _stop_ai_worker(self):
        if self._ai_thread and self._ai_thread.isRunning():
            self._ai_thread.quit(); self._ai_thread.wait(300)
        self._ai_thread = None; self._ai_worker = None; self._ai_busy = False

    def _handle_game_over(self):
        self._ai_timer.stop()
        self._games_played += 1
        wid = self.state.winner
        if wid is not None: self._wins[wid] += 1
        self._total_turns += self.state.turn_number
        avg = self._total_turns / self._games_played
        disp = max(self.human_idx,0)
        self._stats.update_session(self._games_played, self._wins, avg, disp)
        if self.human_idx >= 0:
            self._stats.push_game_result(wid == self.human_idx)
        _stats_manager.record(self._p0_label, self._p1_label, wid, self.state.turn_number)
        if self.human_idx < 0:
            wname = self.state.players[wid].name if wid is not None else "Nobody"
            self._board.set_status(f"Game over – {wname} wins in {self.state.turn_number} turns!")
            self._board.set_interactive(False)
            QTimer.singleShot(1200, self._start_new_game)
        else:
            msg = ("🎉 You Win! Stockpile emptied!" if wid == self.human_idx
                   else "😔 AI wins this round. Better luck next time!")
            self._board.set_status(msg)
            self._board.set_interactive(False)
            r = QMessageBox.question(self,"Game Over",f"{msg}\n\nPlay again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes: self._start_new_game()

    def _confirm_menu(self):
        self._stop_ai_worker(); self._ai_timer.stop()
        r = QMessageBox.question(self,"Return to Menu",
            "Return to main menu? Current game will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes: self.return_to_menu.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Training Screen  ← main focus of this update
# ─────────────────────────────────────────────────────────────────────────────

class TrainingScreen(QWidget):
    return_to_menu    = pyqtSignal()
    training_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        self._session: Optional[TrainingSession] = None
        self._train_thread: Optional[QThread] = None
        self._train_worker: Optional[TrainingWorker] = None
        self._a0_label = "Heuristic"
        self._a1_label = "Random"
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12); root.setSpacing(10)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("🎓  Training Dashboard")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{COLOR_UI_ACCENT};")
        title_row.addWidget(title); title_row.addStretch()
        btn_menu = _mkbtn("← Menu", small=True)
        btn_menu.clicked.connect(self.return_to_menu.emit)
        title_row.addWidget(btn_menu)
        root.addLayout(title_row)

        # ── Top row: config left, progress right ──────────────────────────────
        top = QHBoxLayout(); top.setSpacing(12)

        # Config group
        cfg = _group("Agent Configuration")
        cf  = QVBoxLayout(cfg)
        cs  = f"background:#333344; color:{COLOR_UI_TEXT}; padding:4px; border-radius:4px;"
        ls  = f"color:{COLOR_UI_TEXT}; font-size:11px;"

        # Player 0
        row0 = QHBoxLayout()
        l0 = QLabel("Player 0:"); l0.setStyleSheet(ls); l0.setFixedWidth(68)
        self._a0 = QComboBox(); self._a0.addItems(AGENT_TRAIN_OPTIONS); self._a0.setStyleSheet(cs)
        self._role0 = QLabel(""); self._role0.setStyleSheet("font-size:10px; background:transparent;")
        row0.addWidget(l0); row0.addWidget(self._a0, stretch=1); row0.addWidget(self._role0)
        cf.addLayout(row0)

        # Player 1
        row1 = QHBoxLayout()
        l1 = QLabel("Player 1:"); l1.setStyleSheet(ls); l1.setFixedWidth(68)
        self._a1 = QComboBox(); self._a1.addItems(AGENT_TRAIN_OPTIONS)
        self._a1.setCurrentText("Random"); self._a1.setStyleSheet(cs)
        self._role1 = QLabel(""); self._role1.setStyleSheet("font-size:10px; background:transparent;")
        row1.addWidget(l1); row1.addWidget(self._a1, stretch=1); row1.addWidget(self._role1)
        cf.addLayout(row1)

        # Episodes
        ep_row = QHBoxLayout()
        ep_lbl = QLabel("Episodes:"); ep_lbl.setStyleSheet(ls); ep_lbl.setFixedWidth(68)
        self._ep = QSpinBox(); self._ep.setRange(10, 100_000)
        self._ep.setValue(DEFAULT_TRAIN_EPISODES); self._ep.setStyleSheet(cs)
        ep_row.addWidget(ep_lbl); ep_row.addWidget(self._ep); ep_row.addStretch()
        cf.addLayout(ep_row)

        # Description box
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background:#333344; max-height:1px;")
        cf.addWidget(sep)
        self._desc_lbl = QLabel("")
        self._desc_lbl.setStyleSheet("color:#aaaacc; font-size:10px; background:transparent;")
        self._desc_lbl.setWordWrap(True)
        cf.addWidget(self._desc_lbl)

        # MCTS warning
        self._mcts_warn = QLabel("")
        self._mcts_warn.setStyleSheet("color:#ffaa44; font-size:9px; background:transparent;")
        self._mcts_warn.setWordWrap(True)
        cf.addWidget(self._mcts_warn)

        # Connect signals for live updates
        self._a0.currentTextChanged.connect(self._update_role_labels)
        self._a1.currentTextChanged.connect(self._update_role_labels)
        self._update_role_labels()
        top.addWidget(cfg, stretch=1)

        # Progress group
        prog = _group("Training Progress")
        pl   = QVBoxLayout(prog)

        self._prog_lbl = QLabel("Ready – configure agents above and click Start")
        self._prog_lbl.setStyleSheet(f"color:{COLOR_UI_TEXT}; font-size:11px;")
        pl.addWidget(self._prog_lbl)

        self._pbar = QProgressBar()
        self._pbar.setStyleSheet(f"""
            QProgressBar {{ background:#222233; border:1px solid {COLOR_UI_ACCENT};
                border-radius:4px; text-align:center; color:{COLOR_UI_TEXT}; }}
            QProgressBar::chunk {{ background:{COLOR_UI_ACCENT}; border-radius:4px; }}""")
        pl.addWidget(self._pbar)

        # Chart-generation sub-bar (hidden initially)
        self._chart_lbl = QLabel("Chart generation:")
        self._chart_lbl.setStyleSheet("color:#888899; font-size:9px; background:transparent;")
        self._chart_lbl.setVisible(False)
        pl.addWidget(self._chart_lbl)

        self._chart_pbar = QProgressBar()
        self._chart_pbar.setRange(0,100); self._chart_pbar.setValue(0)
        self._chart_pbar.setMaximumHeight(10)
        self._chart_pbar.setStyleSheet(f"""
            QProgressBar {{ background:#222233; border:1px solid #444455;
                border-radius:4px; }}
            QProgressBar::chunk {{ background:#50c878; border-radius:4px; }}""")
        self._chart_pbar.setVisible(False)
        pl.addWidget(self._chart_pbar)

        self._stat_lbl = QLabel("")
        self._stat_lbl.setStyleSheet("color:#aaaacc; font-size:10px; background:transparent;")
        self._stat_lbl.setWordWrap(True)
        pl.addWidget(self._stat_lbl)

        btn_row = QHBoxLayout()
        self._btn_start = _mkbtn("▶  Start Training", "#2a5a2a", small=True)
        self._btn_start.clicked.connect(self._start)
        self._btn_stop  = _mkbtn("■  Stop", "#5a2a2a", small=True)
        self._btn_stop.clicked.connect(self._stop); self._btn_stop.setEnabled(False)
        self._btn_save  = _mkbtn("💾 Save JSON", "#2a2a5a", small=True)
        self._btn_save.clicked.connect(self._save)
        for b in [self._btn_start, self._btn_stop, self._btn_save]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        pl.addLayout(btn_row)
        top.addWidget(prog, stretch=2)
        root.addLayout(top)

        # Chart area
        chart_grp = _group("Training Results  (computed in background after training)")
        cgl = QVBoxLayout(chart_grp)
        self._chart = EmbeddedChart(figsize=(10, 3.5))
        cgl.addWidget(self._chart)
        root.addWidget(chart_grp, stretch=1)

    # ── Role label updater ─────────────────────────────────────────────────────
    def _update_role_labels(self):
        a0 = self._a0.currentText()
        a1 = self._a1.currentText()

        e0, r0, c0 = _agent_role(a0)
        e1, r1, c1 = _agent_role(a1)

        self._role0.setText(f"{e0} {r0}")
        self._role0.setStyleSheet(f"font-size:10px; color:{c0}; background:transparent;")
        self._role1.setText(f"{e1} {r1}")
        self._role1.setStyleSheet(f"font-size:10px; color:{c1}; background:transparent;")

        self._desc_lbl.setText(self._build_description(a0, a1))

        # MCTS speed warning
        has_mcts = "MCTS" in a0 or "MCTS" in a1
        if has_mcts:
            rollouts = 5
            for t in [a0, a1]:
                if "MCTS" in t and "-" in t:
                    try: rollouts = int(t.split("-")[1])
                    except: pass
            self._mcts_warn.setText(
                f"⚠️ MCTS-{rollouts} training is slow (~{rollouts*0.02:.1f}s/game). "
                "Use MCTS-5 for fastest results.")
        else:
            self._mcts_warn.setText("")

    @staticmethod
    def _build_description(a0: str, a1: str) -> str:
        dqn_p0 = "DQN" in a0
        dqn_p1 = "DQN" in a1

        if dqn_p0 and dqn_p1:
            return ("Both players are DQN agents training against each other "
                    "(self-play). Can produce strong specialists but may develop "
                    "exploitable strategies against each other.")
        elif dqn_p0:
            return (f"Player 0 (DQN) is the LEARNER — it improves with more episodes. "
                    f"Player 1 ({a1}) is the TRAINING OPPONENT. "
                    f"Tip: Heuristic as opponent gives a tougher teacher than Random.")
        elif dqn_p1:
            return (f"Player 1 (DQN) is the LEARNER — it improves with more episodes. "
                    f"Player 0 ({a0}) is the TRAINING OPPONENT. "
                    f"Watch the win rate chart — DQN should improve over time.")
        elif "MCTS" in a0 or "MCTS" in a1:
            return (f"Benchmark run: {a0} vs {a1}. "
                    "Neither agent learns — this measures pure decision-time strength. "
                    "MCTS uses game simulation to pick moves.")
        else:
            return (f"Benchmark run: {a0} vs {a1}. "
                    "Neither agent trains here — results show relative rule/random strength. "
                    "Great for quickly generating statistics for the Stats screen.")

    # ── Training control ───────────────────────────────────────────────────────
    def _start(self):
        self._a0_label = self._a0.currentText()
        self._a1_label = self._a1.currentText()
        n   = self._ep.value()
        a0  = _make_agent(self._a0_label, 0)
        a1  = _make_agent(self._a1_label, 1)
        self._session = TrainingSession(a0, a1, n_episodes=n, seed=0)
        self._pbar.setMaximum(n); self._pbar.setValue(0)
        self._chart_lbl.setVisible(False); self._chart_pbar.setVisible(False)

        self._train_worker = TrainingWorker(self._session)
        self._train_thread = QThread()
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.progress.connect(self._on_progress)
        self._train_worker.finished.connect(self._on_finished)
        # Do NOT call thread.wait() — use deleteLater instead
        self._train_thread.finished.connect(self._train_thread.deleteLater)
        self._train_thread.start()

        self._btn_start.setEnabled(False); self._btn_stop.setEnabled(True)
        self._prog_lbl.setText("Training in progress…")

    def _stop(self):
        if self._train_worker: self._train_worker.stop()
        self._btn_stop.setEnabled(False)
        self._prog_lbl.setText("Stopping…")

    def _on_progress(self, ep, s):
        self._pbar.setValue(ep)
        w = s.get("wins",[0,0]); p = s.get("win_pct",[0,0])
        self._prog_lbl.setText(
            f"Episode {ep} / {self._ep.value()}  —  "
            f"{self._a0_label}: {w[0]} ({p[0]}%)  vs  "
            f"{self._a1_label}: {w[1]} ({p[1]}%)")
        self._stat_lbl.setText(f"Avg turns: {s.get('avg_turns','?')}")

    def _on_finished(self, s):
        self._btn_start.setEnabled(True); self._btn_stop.setEnabled(False)

        w = s.get("wins",[0,0]); p = s.get("win_pct",[0,0])
        self._prog_lbl.setText(
            f"✅ Training complete!  "
            f"{self._a0_label}: {w[0]} ({p[0]}%)  "
            f"{self._a1_label}: {w[1]} ({p[1]}%)")

        # Save results to stats manager
        if self._session:
            for r in self._session.results:
                _stats_manager.record(self._a0_label, self._a1_label,
                                       r["winner"], r["turns"])
        self.training_complete.emit()

        # Generate charts in background with a progress bar
        if self._session and self._session.results:
            self._chart_lbl.setVisible(True)
            self._chart_pbar.setVisible(True)
            self._chart_pbar.setValue(0)
            self._chart_lbl.setText("Computing charts… please wait")

            def _on_chart_done():
                self._chart_lbl.setText("✅ Charts ready!")
                self._chart_pbar.setValue(100)

            self._chart.render_async(
                self._session,
                self._a0_label,
                self._a1_label,
                on_done=_on_chart_done,
                on_progress=self._chart_pbar.setValue,
            )

    def _save(self):
        if not self._session:
            QMessageBox.warning(self,"No Data","Run a session first."); return
        path,_ = QFileDialog.getSaveFileName(self,"Save","training_stats.json","JSON (*.json)")
        if path:
            self._session.save_stats(path)
            QMessageBox.information(self,"Saved",f"Saved to:\n{path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu     = MenuScreen()
        self._game     = GameScreen()
        self._training = TrainingScreen()
        self._stats_v  = StatsViewScreen(stats_manager=_stats_manager)
        self._rules    = RulesScreen()

        for w in [self._menu, self._game, self._training,
                  self._stats_v, self._rules]:
            self._stack.addWidget(w)

        self._menu.human_vs_ai.connect(self._launch_hva)
        self._menu.ai_vs_ai.connect(self._launch_ava)
        self._menu.start_train.connect(lambda: self._stack.setCurrentIndex(2))
        self._menu.view_stats.connect(self._open_stats)
        self._menu.view_rules.connect(lambda: self._stack.setCurrentIndex(4))
        self._menu.exit_app.connect(QApplication.quit)

        self._game.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._training.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._stats_v.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._rules.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._training.training_complete.connect(self._stats_v.refresh)

        self._stack.setCurrentIndex(0)

    def _launch_hva(self):
        dlg = ModeDialog("Human vs AI", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._game.start_human_vs_ai(dlg.ai_combo.currentText())
            self._stack.setCurrentIndex(1)

    def _launch_ava(self):
        dlg = ModeDialog("AI vs AI", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._game.start_ai_vs_ai(dlg.ai0_combo.currentText(),
                                       dlg.ai1_combo.currentText(),
                                       delay=AI_FAST_DELAY_MS)
            self._stack.setCurrentIndex(1)

    def _open_stats(self):
        self._stats_v.refresh()
        self._stack.setCurrentIndex(3)