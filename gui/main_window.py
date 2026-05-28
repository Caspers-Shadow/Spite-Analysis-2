"""
MainWindow – top-level PyQt6 application window for Spite Analysis.
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
)
from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor

from game.game_state import GameState, Action
from game.player import Player
from ai.agent import Agent
from ai.random_agent import RandomAgent
from ai.heuristic_agent import HeuristicAgent
from ai.rl_agent import RLAgent
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

# ── shared stats manager (one instance for the whole app) ─────────────────────
_stats_manager = StatsManager("game_stats.json")


# ─────────────────────────────────────────────────────────────────────────────
# Style helpers
# ─────────────────────────────────────────────────────────────────────────────

BTN = f"""
    QPushButton {{
        background:{COLOR_BUTTON}; color:{COLOR_UI_TEXT}; border:none;
        border-radius:8px; padding:10px 22px;
        font-size:13px; font-weight:bold;
    }}
    QPushButton:hover {{ background:{COLOR_BUTTON_HOVER}; }}
    QPushButton:pressed {{ background:{COLOR_UI_ACCENT}; }}
    QPushButton:disabled {{ background:#333344; color:#555566; }}
"""


def make_btn(text: str, color: Optional[str] = None) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(BTN.replace(COLOR_BUTTON, color, 1) if color else BTN)
    return b


def agent_label(agent: Optional[Agent]) -> str:
    """Human-readable label for an agent (or 'Human')."""
    if agent is None:
        return "Human"
    return type(agent).__name__.replace("Agent", "")


# ─────────────────────────────────────────────────────────────────────────────
# Training worker thread
# ─────────────────────────────────────────────────────────────────────────────

class TrainingWorker(QObject):
    progress = pyqtSignal(int, dict)
    finished = pyqtSignal(dict)

    def __init__(self, session: TrainingSession):
        super().__init__()
        self.session = session
        self._running = True

    def run(self):
        def cb(sess):
            if not self._running:
                sess.stop()
                return
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
    human_vs_ai  = pyqtSignal()
    ai_vs_ai     = pyqtSignal()
    start_train  = pyqtSignal()
    view_stats   = pyqtSignal()
    exit_app     = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(18)

        title = QLabel("♠  SPITE ANALYSIS  ♠")
        title.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{COLOR_UI_ACCENT}; background:transparent; letter-spacing:4px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)

        sub = QLabel("A card game of skill, strategy & spite")
        sub.setFont(QFont("Arial", 12))
        sub.setStyleSheet("color:#888899; background:transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(30)

        for text, sig, color in [
            ("🧑  Human vs AI",      self.human_vs_ai,  "#2a5a2a"),
            ("🤖  AI vs AI",         self.ai_vs_ai,     "#2a2a5a"),
            ("🎓  Start Training",   self.start_train,  "#5a3a1a"),
            ("📊  View Statistics",  self.view_stats,   "#3a1a5a"),
            ("✖   Exit",            self.exit_app,     "#5a1a1a"),
        ]:
            btn = make_btn(text, color)
            btn.setFixedWidth(290)
            btn.clicked.connect(sig.emit)
            lay.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)


# ─────────────────────────────────────────────────────────────────────────────
# Mode config dialog
# ─────────────────────────────────────────────────────────────────────────────

class ModeDialog(QDialog):
    def __init__(self, mode: str, parent=None):
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
            self.ai_combo.addItems(["Heuristic", "Random", "RL (DQN)"])
            self.ai_combo.setStyleSheet(cs)
            l = QLabel("AI Opponent:")
            l.setStyleSheet(ls)
            lay.addRow(l, self.ai_combo)
        else:
            self.ai0_combo = QComboBox()
            self.ai0_combo.addItems(["Heuristic", "Random", "RL (DQN)"])
            self.ai0_combo.setStyleSheet(cs)
            self.ai1_combo = QComboBox()
            self.ai1_combo.addItems(["Random", "Heuristic", "RL (DQN)"])
            self.ai1_combo.setStyleSheet(cs)
            for lbl_text, combo in [("AI Player 1:", self.ai0_combo),
                                      ("AI Player 2:", self.ai1_combo)]:
                l = QLabel(lbl_text)
                l.setStyleSheet(ls)
                lay.addRow(l, combo)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.setStyleSheet(f"color:{COLOR_UI_TEXT};")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)


# ─────────────────────────────────────────────────────────────────────────────
# Game Screen
# ─────────────────────────────────────────────────────────────────────────────

class GameScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: Optional[GameState] = None
        self.agents: List[Optional[Agent]] = [None, None]
        self.human_idx: int = 0       # -1 for AI vs AI
        self.ai_delay: int = AI_THINK_DELAY_MS

        # Labels for StatsManager recording
        self._p0_label: str = "Human"
        self._p1_label: str = "AI"

        # Session counters
        self._games_played = 0
        self._wins: List[int] = [0, 0]
        self._total_turns = 0

        self._ai_timer = QTimer()
        self._ai_timer.timeout.connect(self._ai_step)

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Board
        self._board = BoardView()
        self._board.action_requested.connect(self._on_human_action)

        # Toolbar overlay
        tb = QWidget()
        tb.setStyleSheet("background:rgba(10,10,20,180);")
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(10, 4, 10, 4)

        self._btn_menu = make_btn("← Menu", "#333355")
        self._btn_menu.setFixedWidth(100)
        self._btn_menu.clicked.connect(self._confirm_menu)
        tb_lay.addWidget(self._btn_menu)

        self._btn_new = make_btn("New Game", "#225522")
        self._btn_new.setFixedWidth(120)
        self._btn_new.clicked.connect(self._start_new_game)
        tb_lay.addWidget(self._btn_new)

        tb_lay.addStretch()
        self._mode_lbl = QLabel("")
        self._mode_lbl.setStyleSheet(
            f"color:{COLOR_UI_ACCENT}; font-weight:bold; font-size:11px;")
        tb_lay.addWidget(self._mode_lbl)
        tb_lay.addStretch()

        speed_lbl = QLabel("AI Speed:")
        speed_lbl.setStyleSheet(f"color:{COLOR_UI_TEXT}; font-size:10px;")
        tb_lay.addWidget(speed_lbl)
        self._speed = QSlider(Qt.Orientation.Horizontal)
        self._speed.setRange(0, 4)
        self._speed.setValue(2)
        self._speed.setFixedWidth(100)
        self._speed.valueChanged.connect(self._update_speed)
        tb_lay.addWidget(self._speed)

        board_wrap = QWidget()
        board_wrap.setStyleSheet(f"background:{COLOR_UI_BG};")
        bw_lay = QVBoxLayout(board_wrap)
        bw_lay.setContentsMargins(0, 0, 0, 0)
        bw_lay.setSpacing(0)
        bw_lay.addWidget(tb)
        bw_lay.addWidget(self._board, stretch=1)
        root.addWidget(board_wrap, stretch=1)

        # Stats panel
        self._stats = StatisticsPanel()
        root.addWidget(self._stats)

    # ── Agent creation ─────────────────────────────────────────────────────────
    def _make_agent(self, atype: str, pid: int) -> Agent:
        if atype == "Random":
            return RandomAgent(pid)
        if atype == "RL (DQN)":
            return RLAgent(pid)
        return HeuristicAgent(pid)

    # ── Mode launchers ─────────────────────────────────────────────────────────
    def start_human_vs_ai(self, ai_type: str = "Heuristic"):
        self.human_idx = 0
        self.agents = [None, self._make_agent(ai_type, 1)]
        self._p0_label = "Human"
        self._p1_label = ai_type
        self._mode_lbl.setText(f"Human vs {ai_type}")
        self._reset_session()
        self._start_new_game()

    def start_ai_vs_ai(self, a0: str = "Heuristic", a1: str = "Random",
                       delay: int = AI_FAST_DELAY_MS):
        self.human_idx = -1
        self.agents = [self._make_agent(a0, 0), self._make_agent(a1, 1)]
        self._p0_label = a0
        self._p1_label = a1
        self.ai_delay = delay
        self._mode_lbl.setText(f"{a0} vs {a1}")
        self._reset_session()
        self._start_new_game()

    def _reset_session(self):
        self._games_played = 0
        self._wins = [0, 0]
        self._total_turns = 0

    # ── Game lifecycle ─────────────────────────────────────────────────────────
    def _start_new_game(self):
        self._ai_timer.stop()
        p0name = self._p0_label if self.human_idx != 0 else "Human"
        p1name = self._p1_label if self.human_idx != 1 else "Human"
        self.state = GameState(player_names=(p0name, p1name))

        disp_idx = max(self.human_idx, 0)
        self._board.set_state(self.state, disp_idx)
        self._stats.update_game_state(self.state, disp_idx)

        is_human_turn = self.human_idx >= 0
        self._board.set_interactive(is_human_turn and
                                    self.state.current_player_idx == self.human_idx)
        self._board.set_status(
            "Game started! Select a card to play, or click End Turn to discard."
            if is_human_turn else "AI vs AI – sit back and watch!")
        self._maybe_trigger_ai()

    # ── Human action ───────────────────────────────────────────────────────────
    def _on_human_action(self, action: Action):
        if not self.state or self.state.game_over:
            return
        if self.state.current_player_idx != self.human_idx:
            return

        ok, _ = self.state.apply_action(action)
        if not ok:
            self._board.set_status("❌ Invalid move – try again")
            return

        disp = max(self.human_idx, 0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)

        if self.state.game_over:
            self._handle_game_over()
            return

        if self.state.current_player_idx != self.human_idx:
            self._board.set_interactive(False)
            self._board.set_status("AI is thinking…")
            self._ai_timer.start(self.ai_delay)
        else:
            # Hand was refilled or it's still our turn
            refill_note = ("  (Hand refilled – keep playing!)"
                           if action.action_type.startswith("play") and
                           len(self.state.current_player.hand) == Player.MAX_HAND_SIZE
                           and self.state.hand_refills > 0 else "")
            self._board.set_status(
                "Select a card to play, or click End Turn to discard." + refill_note)

    # ── AI turn ────────────────────────────────────────────────────────────────
    def _maybe_trigger_ai(self):
        if not self.state or self.state.game_over:
            return
        if self.agents[self.state.current_player_idx] is not None:
            self._ai_timer.start(self.ai_delay)

    def _ai_step(self):
        self._ai_timer.stop()
        if not self.state or self.state.game_over:
            return

        pid   = self.state.current_player_idx
        agent = self.agents[pid]
        if agent is None:
            return

        action = agent.choose_action(self.state)
        if action is None:
            log.warning(f"Agent {agent.name} returned None – forcing turn end")
            discards = self.state.get_discard_actions()
            if discards:
                action = discards[0]
            else:
                return

        self.state.apply_action(action)

        disp = max(self.human_idx, 0)
        self._board.set_state(self.state, disp)
        self._stats.update_game_state(self.state, disp)
        self._stats.update_ai_info(agent, str(action))

        if self.state.game_over:
            self._handle_game_over()
            return

        new_pid = self.state.current_player_idx
        if self.agents[new_pid] is not None:
            self._ai_timer.start(self.ai_delay)
        else:
            self._board.set_interactive(True)
            self._board.set_status(
                "Your turn! Select a card to play, or click End Turn to discard.")

    # ── Game over ──────────────────────────────────────────────────────────────
    def _handle_game_over(self):
        self._ai_timer.stop()
        self._games_played += 1
        wid = self.state.winner
        if wid is not None:
            self._wins[wid] += 1
        self._total_turns += self.state.turn_number
        avg = self._total_turns / self._games_played

        # Update side panel session stats
        disp = max(self.human_idx, 0)
        self._stats.update_session(self._games_played, self._wins, avg, disp)

        # Feed the mini win-rate chart
        if self.human_idx >= 0:
            self._stats.push_game_result(wid == self.human_idx)

        # Record in persistent StatsManager
        _stats_manager.record(
            self._p0_label, self._p1_label,
            wid, self.state.turn_number,
        )

        # Determine result message
        if self.human_idx < 0:
            wname = self.state.players[wid].name if wid is not None else "Nobody"
            msg   = f"Game over – {wname} wins in {self.state.turn_number} turns!"
            title = "Game Over"
        elif wid == self.human_idx:
            msg, title = "🎉 You Win!  Stockpile emptied!", "Victory!"
        else:
            msg, title = "😔 AI wins this round.  Better luck next time!", "Defeat"

        self._board.set_status(msg)
        self._board.set_interactive(False)

        if self.human_idx < 0:
            QTimer.singleShot(1200, self._start_new_game)
        else:
            r = QMessageBox.question(
                self, title, f"{msg}\n\nPlay again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes:
                self._start_new_game()

    # ── Speed / nav ────────────────────────────────────────────────────────────
    def _update_speed(self, v: int):
        self.ai_delay = [2000, 1000, 600, 200, 50][v]

    def _confirm_menu(self):
        self._ai_timer.stop()
        r = QMessageBox.question(
            self, "Return to Menu",
            "Return to main menu? Current game will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r == QMessageBox.StandardButton.Yes:
            self.return_to_menu.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Training Screen
# ─────────────────────────────────────────────────────────────────────────────

class TrainingScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        self._session: Optional[TrainingSession] = None
        self._thread:  Optional[QThread] = None
        self._worker:  Optional[TrainingWorker] = None
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 20, 30, 20)
        lay.setSpacing(14)

        title = QLabel("🎓  Training Dashboard")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{COLOR_UI_ACCENT};")
        lay.addWidget(title)

        cfg = QGroupBox("Configuration")
        cfg.setStyleSheet(f"""
            QGroupBox {{ color:{COLOR_UI_ACCENT}; font-weight:bold;
                border:1px solid {COLOR_UI_ACCENT}; border-radius:6px;
                margin-top:6px; background:#2a2a3e; padding:10px; }}
            QGroupBox::title {{ subcontrol-origin:margin; padding:0 4px; }}""")
        form = QFormLayout(cfg)
        cs = f"background:#333344; color:{COLOR_UI_TEXT};"
        ls = f"color:{COLOR_UI_TEXT}; font-size:11px;"

        self._a0 = QComboBox(); self._a0.addItems(["Heuristic","Random","RL (DQN)"])
        self._a0.setStyleSheet(cs)
        self._a1 = QComboBox(); self._a1.addItems(["Random","Heuristic","RL (DQN)"])
        self._a1.setStyleSheet(cs)
        self._ep = QSpinBox(); self._ep.setRange(10, 100000)
        self._ep.setValue(DEFAULT_TRAIN_EPISODES); self._ep.setStyleSheet(cs)

        for txt, w in [("Agent 1:", self._a0), ("Agent 2:", self._a1), ("Episodes:", self._ep)]:
            l = QLabel(txt); l.setStyleSheet(ls)
            form.addRow(l, w)
        lay.addWidget(cfg)

        prog = QGroupBox("Progress")
        prog.setStyleSheet(cfg.styleSheet())
        pl = QVBoxLayout(prog)
        self._prog_lbl = QLabel("Not started")
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
        lay.addWidget(prog)

        row = QHBoxLayout()
        self._btn_start = make_btn("▶  Start", "#2a5a2a")
        self._btn_start.clicked.connect(self._start)
        self._btn_stop  = make_btn("■  Stop",  "#5a2a2a")
        self._btn_stop.clicked.connect(self._stop)
        self._btn_stop.setEnabled(False)
        self._btn_save  = make_btn("💾 Save Stats", "#2a2a5a")
        self._btn_save.clicked.connect(self._save)
        self._btn_plot  = make_btn("📈 Plot",  "#3a2a5a")
        self._btn_plot.clicked.connect(self._plot)
        self._btn_menu  = make_btn("← Menu")
        self._btn_menu.clicked.connect(self.return_to_menu.emit)
        for b in [self._btn_start, self._btn_stop, self._btn_save,
                  self._btn_plot, self._btn_menu]:
            row.addWidget(b)
        lay.addLayout(row)
        lay.addStretch()

    def _make_agent(self, t, pid):
        if t == "Random":    return RandomAgent(pid)
        if t == "RL (DQN)":  return RLAgent(pid)
        return HeuristicAgent(pid)

    def _start(self):
        n  = self._ep.value()
        a0 = self._make_agent(self._a0.currentText(), 0)
        a1 = self._make_agent(self._a1.currentText(), 1)
        self._session = TrainingSession(a0, a1, n_episodes=n, seed=0)
        self._pbar.setMaximum(n); self._pbar.setValue(0)

        self._thread = QThread()
        self._worker = TrainingWorker(self._session)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._prog_lbl.setText("Training…")

    def _stop(self):
        if self._worker: self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_progress(self, ep, s):
        self._pbar.setValue(ep)
        self._prog_lbl.setText(f"Episode {ep} / {self._ep.value()}")
        w = s.get("wins", [0,0]); p = s.get("win_pct", [0,0])
        self._stat_lbl.setText(
            f"Wins: {w[0]} ({p[0]}%) vs {w[1]} ({p[1]}%)  |  "
            f"Avg turns: {s.get('avg_turns','?')}")

    def _on_finished(self, s):
        self._btn_start.setEnabled(True); self._btn_stop.setEnabled(False)
        self._prog_lbl.setText("Training complete!")
        w = s.get("wins",[0,0]); p = s.get("win_pct",[0,0])
        self._stat_lbl.setText(
            f"Final – Wins: {w[0]} ({p[0]}%) vs {w[1]} ({p[1]}%)  |  "
            f"Avg turns: {s.get('avg_turns','?')}")

        # Persist results to StatsManager
        a0lbl = self._a0.currentText()
        a1lbl = self._a1.currentText()
        if self._session:
            for r in self._session.results:
                _stats_manager.record(a0lbl, a1lbl, r["winner"], r["turns"])

        if self._thread:
            self._thread.quit(); self._thread.wait()

    def _save(self):
        if not self._session:
            QMessageBox.warning(self,"No Data","Run a session first."); return
        path, _ = QFileDialog.getSaveFileName(self,"Save Stats","training_stats.json","JSON (*.json)")
        if path:
            self._session.save_stats(path)
            QMessageBox.information(self,"Saved",f"Stats saved to:\n{path}")

    def _plot(self):
        if not self._session:
            QMessageBox.warning(self,"No Data","Run a session first."); return
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        self._session.plot_stats()
        # Use non-blocking show so Qt event loop stays alive
        plt.show(block=False)


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
            self._stack.addWidget(w)   # indices 0-3

        self._menu.human_vs_ai.connect(self._launch_hva)
        self._menu.ai_vs_ai.connect(self._launch_ava)
        self._menu.start_train.connect(lambda: self._stack.setCurrentIndex(2))
        self._menu.view_stats.connect(self._open_stats)
        self._menu.exit_app.connect(QApplication.quit)

        self._game.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._training.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._stats_v.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))

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
        self._stats_v.refresh()   # pick up any newly recorded games
        self._stack.setCurrentIndex(3)