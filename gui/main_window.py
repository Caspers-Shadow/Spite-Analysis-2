"""
MainWindow – top-level application window for Spite Analysis.

Changes vs previous version
----------------------------
* TrainingScreen shows embedded matplotlib plots immediately after training
  finishes (no plt.show() / event-loop conflicts).
* Training results are pushed into _stats_manager automatically on finish.
* StatsViewScreen is purely memory-driven – no file-load needed.
* MainWindow._open_stats() always refreshes the view from current memory.
"""

import json, os, sys
from typing import Optional, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox,
    QMessageBox, QFileDialog, QSlider,
    QGroupBox, QDialog, QDialogButtonBox,
    QFormLayout, QApplication, QProgressBar,
    QSplitter, QScrollArea,
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

# ── App-wide stats manager (in-memory + optional JSON backup) ─────────────────
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
SMALL_BTN = BTN.replace("13px", "11px").replace("10px 22px", "6px 14px").replace("8px", "6px")

def _mkbtn(text, color=None, small=False):
    b = QPushButton(text)
    style = SMALL_BTN if small else BTN
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

def agent_label(agent):
    return "Human" if agent is None else type(agent).__name__.replace("Agent","")


# ─────────────────────────────────────────────────────────────────────────────
# Embedded chart widget (used in training screen)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddedChart(QWidget):
    """A matplotlib Figure embedded in a QWidget. Redrawn by calling plot()."""
    def __init__(self, figsize=(10,4), parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:#1e1e2e;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        if _MPL:
            self._fig = Figure(figsize=figsize, facecolor="#1e1e2e", tight_layout=True)
            self._canvas = FigureCanvas(self._fig)
            layout.addWidget(self._canvas)
        else:
            layout.addWidget(_lbl("matplotlib not installed", color="#ff7f7f"))
        self._show_waiting()

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

    def plot_training(self, session: TrainingSession, a0_label: str, a1_label: str):
        """Draw 4-panel training summary into the embedded figure."""
        if not _MPL or not session.results: return
        BG, PANEL = "#1e1e2e", "#2a2a3e"
        ACC, GRN, SAL, GRY = "#7c6af7", "#50c878", "#ff7f7f", "#888899"

        self._fig.clf()
        axes = self._fig.subplots(1, 3)

        results = session.results
        turns   = [r["turns"] for r in results]

        # 1. Rolling win rate
        window = min(50, max(10, len(results)//10))
        xs, ys = [], []
        for i in range(1, len(results)+1):
            chunk = results[max(0,i-window):i]
            xs.append(i)
            ys.append(sum(1 for r in chunk if r["winner"]==0)/len(chunk))

        ax = axes[0]
        ax.set_facecolor(PANEL)
        ax.plot(xs, ys, color=ACC, linewidth=2)
        ax.axhline(0.5, color=GRY, linestyle="--", linewidth=1)
        ax.fill_between(xs, ys, 0.5, where=[y>0.5 for y in ys], alpha=0.2, color=GRN)
        ax.fill_between(xs, ys, 0.5, where=[y<=0.5 for y in ys], alpha=0.2, color=SAL)
        ax.set_ylim(0,1); ax.set_title(f"{a0_label} Win Rate (roll {window})",
                                        color=COLOR_UI_TEXT, fontsize=9)
        ax.set_xlabel("Episode", color=GRY, fontsize=8)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v,_: f"{v:.0%}"))
        for s in ax.spines.values(): s.set_color("#333344")
        ax.tick_params(colors=GRY, labelsize=7)

        # 2. Game length over time (smoothed)
        smooth = [sum(turns[max(0,i-20):i+1])/min(i+1,20) for i in range(len(turns))]
        ax2 = axes[1]
        ax2.set_facecolor(PANEL)
        ax2.plot(range(len(smooth)), smooth, color=GRN, linewidth=1.5)
        ax2.set_title("Avg Game Length (rolling 20)", color=COLOR_UI_TEXT, fontsize=9)
        ax2.set_xlabel("Episode", color=GRY, fontsize=8)
        ax2.set_ylabel("Turns", color=GRY, fontsize=8)
        for s in ax2.spines.values(): s.set_color("#333344")
        ax2.tick_params(colors=GRY, labelsize=7)
        ax2.grid(True, alpha=0.15, color=GRY)

        # 3. Win bar
        w = session.wins
        draws = len(results) - w[0] - w[1]
        ax3 = axes[2]
        ax3.set_facecolor(PANEL)
        labels = [a0_label, a1_label, "Draw"]
        vals   = [w[0], w[1], draws]
        colors = [ACC, SAL, GRY]
        bars = ax3.bar(labels, vals, color=colors, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(val), ha="center", color=COLOR_UI_TEXT, fontsize=9)
        total = len(results)
        ax3.set_title(f"Results ({total} games)", color=COLOR_UI_TEXT, fontsize=9)
        ax3.set_ylabel("Games", color=GRY, fontsize=8)
        for s in ax3.spines.values(): s.set_color("#333344")
        ax3.tick_params(colors=GRY, labelsize=8)
        ax3.grid(True, alpha=0.15, axis="y", color=GRY)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# Training worker thread
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main Menu
# ─────────────────────────────────────────────────────────────────────────────

class MenuScreen(QWidget):
    human_vs_ai = pyqtSignal()
    ai_vs_ai    = pyqtSignal()
    start_train = pyqtSignal()
    view_stats  = pyqtSignal()
    exit_app    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(18)

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
        lay.addSpacing(30)

        for text, sig, color in [
            ("🧑  Human vs AI",     self.human_vs_ai,  "#2a5a2a"),
            ("🤖  AI vs AI",        self.ai_vs_ai,     "#2a2a5a"),
            ("🎓  Start Training",  self.start_train,  "#5a3a1a"),
            ("📊  View Statistics", self.view_stats,   "#3a1a5a"),
            ("✖   Exit",           self.exit_app,     "#5a1a1a"),
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
        self.setMinimumWidth(320)
        lay = QFormLayout(self)
        lay.setSpacing(10)
        cs = f"background:#333344; color:{COLOR_UI_TEXT}; padding:4px;"
        ls = f"color:{COLOR_UI_TEXT}; font-size:11px;"

        if mode == "Human vs AI":
            self.ai_combo = QComboBox()
            self.ai_combo.addItems(["Heuristic","MCTS-50","Random","RL (DQN)"])
            self.ai_combo.setStyleSheet(cs)
            l = QLabel("AI Opponent:"); l.setStyleSheet(ls)
            lay.addRow(l, self.ai_combo)
        else:
            self.ai0_combo = QComboBox()
            self.ai0_combo.addItems(["Heuristic","MCTS-50","Random","RL (DQN)"])
            self.ai0_combo.setStyleSheet(cs)
            self.ai1_combo = QComboBox()
            self.ai1_combo.addItems(["Random","Heuristic","MCTS-50","RL (DQN)"])
            self.ai1_combo.setStyleSheet(cs)
            for txt, w in [("AI Player 1:", self.ai0_combo), ("AI Player 2:", self.ai1_combo)]:
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
        self.ai_delay: int = AI_THINK_DELAY_MS
        self._p0_label = "Human"
        self._p1_label = "AI"
        self._games_played = 0
        self._wins: List[int] = [0, 0]
        self._total_turns = 0
        self._ai_timer = QTimer()
        self._ai_timer.timeout.connect(self._ai_step)
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        self._board = BoardView()
        self._board.action_requested.connect(self._on_human_action)

        tb = QWidget(); tb.setStyleSheet("background:rgba(10,10,20,180);")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(10,4,10,4)
        self._btn_menu = _mkbtn("← Menu","#333355",small=True); self._btn_menu.setFixedWidth(100)
        self._btn_menu.clicked.connect(self._confirm_menu)
        self._btn_new = _mkbtn("New Game","#225522",small=True); self._btn_new.setFixedWidth(120)
        self._btn_new.clicked.connect(self._start_new_game)
        tbl.addWidget(self._btn_menu); tbl.addWidget(self._btn_new); tbl.addStretch()
        self._mode_lbl = QLabel("")
        self._mode_lbl.setStyleSheet(f"color:{COLOR_UI_ACCENT}; font-weight:bold; font-size:11px;")
        tbl.addWidget(self._mode_lbl); tbl.addStretch()
        tbl.addWidget(QLabel("AI Speed:"))
        self._speed = QSlider(Qt.Orientation.Horizontal)
        self._speed.setRange(0,4); self._speed.setValue(2); self._speed.setFixedWidth(100)
        self._speed.valueChanged.connect(lambda v: setattr(self,'ai_delay',[2000,1000,600,200,50][v]))
        tbl.addWidget(self._speed)

        bw = QWidget(); bw.setStyleSheet(f"background:{COLOR_UI_BG};")
        bwl = QVBoxLayout(bw); bwl.setContentsMargins(0,0,0,0); bwl.setSpacing(0)
        bwl.addWidget(tb); bwl.addWidget(self._board, stretch=1)
        root.addWidget(bw, stretch=1)

        self._stats = StatisticsPanel()
        root.addWidget(self._stats)

    def _make_agent(self, t, pid):
        if t=="Random":         return RandomAgent(pid)
        if t=="RL (DQN)":       return RLAgent(pid)
        if t.startswith("MCTS"):
            n = int(t.split("-")[1]) if "-" in t else 50
            return MCTSAgent(pid, rollouts=n)
        return HeuristicAgent(pid)

    def start_human_vs_ai(self, ai_type="Heuristic"):
        self.human_idx = 0
        self.agents = [None, self._make_agent(ai_type,1)]
        self._p0_label = "Human"; self._p1_label = ai_type
        self._mode_lbl.setText(f"Human vs {ai_type}")
        self._games_played=0; self._wins=[0,0]; self._total_turns=0
        self._start_new_game()

    def start_ai_vs_ai(self, a0="Heuristic", a1="Random", delay=AI_FAST_DELAY_MS):
        self.human_idx = -1
        self.agents = [self._make_agent(a0,0), self._make_agent(a1,1)]
        self._p0_label=a0; self._p1_label=a1; self.ai_delay=delay
        self._mode_lbl.setText(f"{a0} vs {a1}")
        self._games_played=0; self._wins=[0,0]; self._total_turns=0
        self._start_new_game()

    def _start_new_game(self):
        self._ai_timer.stop()
        p0n = self._p0_label if self.human_idx!=0 else "Human"
        p1n = self._p1_label if self.human_idx!=1 else "Human"
        self.state = GameState(player_names=(p0n, p1n))
        disp = max(self.human_idx,0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        self._board.set_interactive(self.human_idx>=0 and
                                    self.state.current_player_idx==self.human_idx)
        self._board.set_status(
            "Select a card to play, or click End Turn to discard & end your turn."
            if self.human_idx>=0 else "AI vs AI – watching…")
        self._maybe_trigger_ai()

    def _on_human_action(self, action: Action):
        if not self.state or self.state.game_over: return
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
            extra = ("  ✨ Fresh hand drawn – keep playing!"
                     if action.action_type.startswith("play") and
                        self.state.hand_refills > 0 and
                        len(self.state.current_player.hand) == Player.MAX_HAND_SIZE
                     else "")
            self._board.set_status("Play a card or click End Turn to discard." + extra)

    def _maybe_trigger_ai(self):
        if not self.state or self.state.game_over: return
        if self.agents[self.state.current_player_idx] is not None:
            self._ai_timer.start(self.ai_delay)

    def _ai_step(self):
        self._ai_timer.stop()
        if not self.state or self.state.game_over: return
        pid = self.state.current_player_idx
        agent = self.agents[pid]
        if agent is None: return
        action = agent.choose_action(self.state)
        if action is None:
            log.warning(f"Agent {agent.name} returned None")
            discards = self.state.get_discard_actions()
            if discards: action = discards[0]
            else: return
        self.state.apply_action(action)
        disp = max(self.human_idx,0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        self._stats.update_ai_info(agent, str(action))
        if self.state.game_over:
            self._handle_game_over(); return
        new_pid = self.state.current_player_idx
        if self.agents[new_pid] is not None:
            self._ai_timer.start(self.ai_delay)
        else:
            self._board.set_interactive(True)
            self._board.set_status("Your turn! Play a card or click End Turn.")

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
        _stats_manager.record(self._p0_label, self._p1_label,
                               wid, self.state.turn_number)
        if self.human_idx < 0:
            wname = self.state.players[wid].name if wid is not None else "Nobody"
            msg = f"Game over – {wname} wins in {self.state.turn_number} turns!"
            self._board.set_status(msg)
            self._board.set_interactive(False)
            QTimer.singleShot(1200, self._start_new_game)
        else:
            msg = ("🎉 You Win! Stockpile emptied!" if wid==self.human_idx
                   else "😔 AI wins this round. Better luck next time!")
            self._board.set_status(msg)
            self._board.set_interactive(False)
            r = QMessageBox.question(self, "Game Over", f"{msg}\n\nPlay again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self._start_new_game()

    def _confirm_menu(self):
        self._ai_timer.stop()
        r = QMessageBox.question(self,"Return to Menu",
            "Return to main menu? Current game will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes:
            self.return_to_menu.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Training Screen  (with embedded live charts)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingScreen(QWidget):
    return_to_menu = pyqtSignal()
    # Emitted when training finishes so MainWindow can refresh stats view
    training_complete = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        self._session: Optional[TrainingSession] = None
        self._thread:  Optional[QThread] = None
        self._worker:  Optional[TrainingWorker] = None
        self._a0_label = "Heuristic"
        self._a1_label = "Random"
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

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

        # Top: config + progress side by side
        top = QHBoxLayout(); top.setSpacing(12)

        # Config
        cfg = _group("Configuration")
        cf = QFormLayout(cfg)
        cs = f"background:#333344; color:{COLOR_UI_TEXT}; padding:4px;"
        ls = f"color:{COLOR_UI_TEXT}; font-size:11px;"
        self._a0 = QComboBox(); self._a0.addItems(["Heuristic","MCTS-50","Random","RL (DQN)"]); self._a0.setStyleSheet(cs)
        self._a1 = QComboBox(); self._a1.addItems(["Random","Heuristic","MCTS-50","RL (DQN)"]); self._a1.setStyleSheet(cs)
        self._ep = QSpinBox(); self._ep.setRange(10,100000); self._ep.setValue(DEFAULT_TRAIN_EPISODES); self._ep.setStyleSheet(cs)
        for txt,w in [("Agent 1 (P0):",self._a0),("Agent 2 (P1):",self._a1),("Episodes:",self._ep)]:
            l=QLabel(txt); l.setStyleSheet(ls); cf.addRow(l,w)
        top.addWidget(cfg, stretch=1)

        # Progress
        prog = _group("Progress")
        pl = QVBoxLayout(prog)
        self._prog_lbl = QLabel("Ready – configure and click Start Training")
        self._prog_lbl.setStyleSheet(f"color:{COLOR_UI_TEXT}; font-size:11px;")
        pl.addWidget(self._prog_lbl)
        self._pbar = QProgressBar()
        self._pbar.setStyleSheet(f"""
            QProgressBar {{ background:#222233; border:1px solid {COLOR_UI_ACCENT};
                border-radius:4px; text-align:center; color:{COLOR_UI_TEXT}; }}
            QProgressBar::chunk {{ background:{COLOR_UI_ACCENT}; border-radius:4px; }}""")
        pl.addWidget(self._pbar)
        self._stat_lbl = QLabel("")
        self._stat_lbl.setStyleSheet("color:#aaaacc; font-size:10px;")
        self._stat_lbl.setWordWrap(True)
        pl.addWidget(self._stat_lbl)

        # Buttons row inside progress box
        btn_row = QHBoxLayout()
        self._btn_start = _mkbtn("▶  Start Training","#2a5a2a",small=True)
        self._btn_start.clicked.connect(self._start)
        self._btn_stop = _mkbtn("■  Stop","#5a2a2a",small=True)
        self._btn_stop.clicked.connect(self._stop); self._btn_stop.setEnabled(False)
        self._btn_save = _mkbtn("💾 Save JSON","#2a2a5a",small=True)
        self._btn_save.clicked.connect(self._save)
        for b in [self._btn_start, self._btn_stop, self._btn_save]:
            btn_row.addWidget(b)
        btn_row.addStretch()
        pl.addLayout(btn_row)
        top.addWidget(prog, stretch=2)
        root.addLayout(top)

        # Embedded chart (fills remaining space)
        chart_grp = _group("Training Results  (auto-updates when training finishes)")
        cgl = QVBoxLayout(chart_grp)
        self._chart = EmbeddedChart(figsize=(10,3.5))
        cgl.addWidget(self._chart)
        root.addWidget(chart_grp, stretch=1)

    def _make_agent(self, t, pid):
        if t=="Random":         return RandomAgent(pid)
        if t=="RL (DQN)":       return RLAgent(pid)
        if t.startswith("MCTS"):
            n = int(t.split("-")[1]) if "-" in t else 50
            return MCTSAgent(pid, rollouts=n)
        return HeuristicAgent(pid)

    def _start(self):
        self._a0_label = self._a0.currentText()
        self._a1_label = self._a1.currentText()
        n = self._ep.value()
        a0 = self._make_agent(self._a0_label, 0)
        a1 = self._make_agent(self._a1_label, 1)
        self._session = TrainingSession(a0, a1, n_episodes=n, seed=0)
        self._pbar.setMaximum(n); self._pbar.setValue(0)

        self._thread = QThread()
        self._worker = TrainingWorker(self._session)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self._btn_start.setEnabled(False); self._btn_stop.setEnabled(True)
        self._prog_lbl.setText("Training in progress…")

    def _stop(self):
        if self._worker: self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_progress(self, ep, s):
        self._pbar.setValue(ep)
        self._prog_lbl.setText(f"Episode {ep} / {self._ep.value()}")
        w=s.get("wins",[0,0]); p=s.get("win_pct",[0,0])
        self._stat_lbl.setText(
            f"{self._a0_label}: {w[0]} wins ({p[0]}%)  |  "
            f"{self._a1_label}: {w[1]} wins ({p[1]}%)  |  "
            f"Avg turns: {s.get('avg_turns','?')}")

    def _on_finished(self, s):
        self._btn_start.setEnabled(True); self._btn_stop.setEnabled(False)
        w=s.get("wins",[0,0]); p=s.get("win_pct",[0,0])
        self._prog_lbl.setText(
            f"✅ Training complete!  "
            f"{self._a0_label}: {w[0]} ({p[0]}%)  "
            f"{self._a1_label}: {w[1]} ({p[1]}%)")

        # 1. Draw embedded charts immediately
        if self._session:
            self._chart.plot_training(self._session, self._a0_label, self._a1_label)

        # 2. Push all results into the global stats manager
        if self._session:
            for r in self._session.results:
                _stats_manager.record(self._a0_label, self._a1_label,
                                      r["winner"], r["turns"])

        # 3. Signal main window to refresh stats view
        self.training_complete.emit()

        if self._thread:
            self._thread.quit(); self._thread.wait()

    def _save(self):
        if not self._session:
            QMessageBox.warning(self,"No Data","Run a training session first."); return
        path,_ = QFileDialog.getSaveFileName(
            self,"Save Stats","training_stats.json","JSON (*.json)")
        if path:
            self._session.save_stats(path)
            QMessageBox.information(self,"Saved",f"Stats saved to:\n{path}")


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

        for w in [self._menu, self._game, self._training, self._stats_v]:
            self._stack.addWidget(w)

        self._menu.human_vs_ai.connect(self._launch_hva)
        self._menu.ai_vs_ai.connect(self._launch_ava)
        self._menu.start_train.connect(lambda: self._stack.setCurrentIndex(2))
        self._menu.view_stats.connect(self._open_stats)
        self._menu.exit_app.connect(QApplication.quit)

        self._game.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._training.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._stats_v.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))

        # Auto-refresh stats view whenever training completes
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
        self._stats_v.refresh()   # always pull latest in-memory data
        self._stack.setCurrentIndex(3)