"""
MainWindow – top-level PyQt6 application window for Spite Analysis.

Manages:
  - Main menu screen
  - Human vs AI gameplay
  - AI vs AI self-play / training
  - Statistics export
  - AI turn timers
"""

import json
import os
import sys
from typing import Optional, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox,
    QMessageBox, QFileDialog, QSlider,
    QGroupBox, QCheckBox, QDialog,
    QDialogButtonBox, QFormLayout, QApplication,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

from game.game_state import GameState, Action
from game.player import Player
from ai.agent import Agent
from ai.random_agent import RandomAgent
from ai.heuristic_agent import HeuristicAgent
from ai.rl_agent import RLAgent
from ai.training import TrainingSession, run_game
from gui.board_view import BoardView
from gui.statistics_panel import StatisticsPanel
from utils.config import (
    WINDOW_TITLE, WINDOW_MIN_W, WINDOW_MIN_H,
    AI_THINK_DELAY_MS, AI_FAST_DELAY_MS,
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
    DEFAULT_TRAIN_EPISODES,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BTN_STYLE = f"""
    QPushButton {{
        background: {COLOR_BUTTON};
        color: {COLOR_UI_TEXT};
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background: {COLOR_BUTTON_HOVER};
    }}
    QPushButton:pressed {{
        background: {COLOR_UI_ACCENT};
    }}
    QPushButton:disabled {{
        background: #333344;
        color: #555566;
    }}
"""


def make_btn(text: str, color: Optional[str] = None) -> QPushButton:
    b = QPushButton(text)
    style = BTN_STYLE
    if color:
        style = style.replace(COLOR_BUTTON, color, 1)
    b.setStyleSheet(style)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Training worker thread
# ─────────────────────────────────────────────────────────────────────────────

class TrainingWorker(QObject):
    """Runs TrainingSession in a background thread."""

    progress = pyqtSignal(int, dict)     # (episode, summary)
    finished = pyqtSignal(dict)

    def __init__(self, session: TrainingSession):
        super().__init__()
        self.session = session
        self._running = True

    def run(self):
        def callback(sess: TrainingSession):
            if not self._running:
                sess.stop()
                return
            if sess.episode % 10 == 0:
                self.progress.emit(sess.episode, sess.get_summary())

        self.session.run(callback=callback)
        self.finished.emit(self.session.get_summary())

    def stop(self):
        self._running = False
        self.session.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Main Menu Screen
# ─────────────────────────────────────────────────────────────────────────────

class MenuScreen(QWidget):
    human_vs_ai  = pyqtSignal()
    ai_vs_ai     = pyqtSignal()
    start_train  = pyqtSignal()
    view_stats   = pyqtSignal()
    exit_app     = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        # Title
        title = QLabel("♠ SPITE ANALYSIS ♠")
        title.setFont(QFont("Arial", 36, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_UI_ACCENT}; background: transparent; letter-spacing: 4px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("A card game of skill, strategy & spite")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setStyleSheet(f"color: #888899; background: transparent;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        buttons = [
            ("🧑 Human vs AI",     self.human_vs_ai,  "#2a5a2a"),
            ("🤖 AI vs AI",        self.ai_vs_ai,     "#2a2a5a"),
            ("🎓 Start Training",  self.start_train,  "#5a3a1a"),
            ("📊 View Statistics", self.view_stats,   "#3a1a5a"),
            ("✖  Exit",           self.exit_app,     "#5a1a1a"),
        ]
        for text, signal, color in buttons:
            btn = make_btn(text, color)
            btn.setFixedWidth(280)
            btn.clicked.connect(signal.emit)
            layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignCenter)


# ─────────────────────────────────────────────────────────────────────────────
# Game Screen
# ─────────────────────────────────────────────────────────────────────────────

class GameScreen(QWidget):
    """
    The main gameplay screen.  Works for both Human vs AI and AI vs AI.
    """
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: Optional[GameState] = None
        self.agents: List[Optional[Agent]] = [None, None]
        self.human_idx: int = 0          # -1 for AI vs AI
        self.ai_delay: int = AI_THINK_DELAY_MS

        # Session stats
        self._games_played: int = 0
        self._wins: List[int] = [0, 0]
        self._total_turns: int = 0

        self._ai_timer = QTimer()
        self._ai_timer.timeout.connect(self._ai_step)

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background: {COLOR_UI_BG};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Board
        self._board = BoardView()
        self._board.action_requested.connect(self._on_human_action)
        root.addWidget(self._board, stretch=1)

        # Side panel
        self._stats = StatisticsPanel()
        self._stats.end_turn_requested.connect(self._on_end_turn_requested)
        root.addWidget(self._stats)

        # Top toolbar (overlaid via separate layout)
        toolbar = QWidget(self)
        toolbar.setStyleSheet("background: rgba(10,10,20,180); border-radius: 0px;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 4, 10, 4)

        self._btn_menu = make_btn("← Menu", "#333355")
        self._btn_menu.setFixedWidth(100)
        self._btn_menu.clicked.connect(self._confirm_menu)
        tb_layout.addWidget(self._btn_menu)

        self._btn_new_game = make_btn("New Game", "#225522")
        self._btn_new_game.setFixedWidth(120)
        self._btn_new_game.clicked.connect(self._start_new_game)
        tb_layout.addWidget(self._btn_new_game)

        tb_layout.addStretch()

        self._mode_label = QLabel("Mode: Human vs AI")
        self._mode_label.setStyleSheet(f"color: {COLOR_UI_ACCENT}; font-weight: bold; font-size: 11px;")
        tb_layout.addWidget(self._mode_label)

        tb_layout.addStretch()

        speed_lbl = QLabel("AI Speed:")
        speed_lbl.setStyleSheet(f"color: {COLOR_UI_TEXT}; font-size: 10px;")
        tb_layout.addWidget(speed_lbl)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, 4)
        self._speed_slider.setValue(2)
        self._speed_slider.setFixedWidth(100)
        self._speed_slider.valueChanged.connect(self._update_speed)
        tb_layout.addWidget(self._speed_slider)

        # Inject toolbar into board area
        self._board.layout().insertWidget(0, toolbar)

    # ── Setup ──────────────────────────────────────────────────────────────────
    def start_human_vs_ai(self, ai_type: str = "Heuristic"):
        self.human_idx = 0
        self.agents[0] = None   # Human
        self.agents[1] = self._create_agent(ai_type, player_id=1)
        self._mode_label.setText(f"Human vs {ai_type} AI")
        self._start_new_game()

    def start_ai_vs_ai(self, ai0_type: str = "Heuristic", ai1_type: str = "Random",
                       delay: int = AI_FAST_DELAY_MS):
        self.human_idx = -1
        self.agents[0] = self._create_agent(ai0_type, player_id=0)
        self.agents[1] = self._create_agent(ai1_type, player_id=1)
        self.ai_delay = delay
        self._mode_label.setText(f"AI vs AI: {ai0_type} vs {ai1_type}")
        self._start_new_game()

    def _create_agent(self, ai_type: str, player_id: int) -> Agent:
        if ai_type == "Random":
            return RandomAgent(player_id)
        elif ai_type == "RL (DQN)":
            return RLAgent(player_id)
        else:
            return HeuristicAgent(player_id)

    def _start_new_game(self):
        self._ai_timer.stop()
        pnames = ("Human", "AI") if self.human_idx == 0 else \
                 (self.agents[0].name if self.agents[0] else "AI 1",
                  self.agents[1].name if self.agents[1] else "AI 2")
        self.state = GameState(player_names=pnames)
        self._board.set_state(self.state, max(self.human_idx, 0))
        self._stats.update_game_state(self.state, max(self.human_idx, 0))
        self._board.set_status("Game started! It's your turn." if self.human_idx == 0
                               else "AI vs AI – watch and enjoy!")
        self._board.set_interactive(self.human_idx >= 0 and
                                    self.state.current_player_idx == self.human_idx)
        self._maybe_trigger_ai()

    # ── Human actions ──────────────────────────────────────────────────────────
    def _on_human_action(self, action: Action):
        if not self.state or self.state.game_over:
            return
        if self.state.current_player_idx != self.human_idx:
            return

        success, winner = self.state.apply_action(action)
        if not success:
            self._board.set_status("❌ Invalid move – try again")
            return

        self._board.clear_selection()
        self._board.set_state(self.state, self.human_idx)
        self._stats.update_game_state(self.state, self.human_idx)

        if self.state.game_over:
            self._handle_game_over()
            return

        if self.state.current_player_idx != self.human_idx:
            # AI's turn
            self._board.set_interactive(False)
            self._board.set_status("AI is thinking…")
            self._ai_timer.start(self.ai_delay)
        else:
            phase_msg = "Play a card, or click a discard pile to end your turn." \
                        if self.state.phase == GameState.PHASE_PLAY \
                        else "Select a hand card then click a discard pile."
            self._board.set_status(phase_msg)

    def _on_end_turn_requested(self):
        """Human clicks 'End Turn' – auto-discard the least valuable hand card."""
        if not self.state or self.state.game_over:
            return
        if self.state.current_player_idx != self.human_idx:
            return
        discards = self.state.get_discard_actions()
        if discards:
            # Pick the first available discard
            self._on_human_action(discards[0])
        else:
            self._board.set_status("No cards to discard!")

    # ── AI turn loop ───────────────────────────────────────────────────────────
    def _maybe_trigger_ai(self):
        if not self.state or self.state.game_over:
            return
        pid = self.state.current_player_idx
        if self.agents[pid] is not None:
            self._ai_timer.start(self.ai_delay)

    def _ai_step(self):
        self._ai_timer.stop()
        if not self.state or self.state.game_over:
            return

        pid = self.state.current_player_idx
        agent = self.agents[pid]
        if agent is None:
            return

        action = agent.choose_action(self.state)
        if action is None:
            log.warning(f"AI {agent.name} returned None – ending turn")
            return

        success, winner = self.state.apply_action(action)
        if not success:
            log.error(f"AI action failed: {action}")

        human_disp = max(self.human_idx, 0)
        self._board.set_state(self.state, human_disp)
        self._stats.update_game_state(self.state, human_disp)
        self._stats.update_ai_info(agent, str(action))

        if self.state.game_over:
            self._handle_game_over()
            return

        new_pid = self.state.current_player_idx
        if self.agents[new_pid] is not None:
            # Continue AI turn
            self._ai_timer.start(self.ai_delay)
        else:
            # Human's turn
            self._board.set_interactive(True)
            phase_msg = ("Play cards or click a discard pile to end your turn."
                         if self.state.phase == GameState.PHASE_PLAY
                         else "Select a hand card then click a discard pile.")
            self._board.set_status(phase_msg)

    # ── Game over ──────────────────────────────────────────────────────────────
    def _handle_game_over(self):
        self._ai_timer.stop()
        self._games_played += 1
        winner_id = self.state.winner
        if winner_id is not None:
            self._wins[winner_id] += 1
        self._total_turns += self.state.turn_number

        avg_turns = self._total_turns / self._games_played
        self._stats.update_session_stats(
            self._games_played, self._wins, avg_turns, max(self.human_idx, 0)
        )

        if winner_id == self.human_idx:
            msg = "🎉 You Win!  Stockpile emptied!"
            title = "Victory!"
        elif self.human_idx < 0:
            winner_name = self.state.players[winner_id].name if winner_id is not None else "No one"
            msg = f"Game Over – {winner_name} wins in {self.state.turn_number} turns!"
            title = "Game Over"
        else:
            msg = f"😔 AI wins this round.  Better luck next time!"
            title = "Defeat"

        self._board.set_status(msg)
        self._board.set_interactive(False)

        # Auto-restart for AI vs AI
        if self.human_idx < 0:
            QTimer.singleShot(1200, self._start_new_game)
        else:
            reply = QMessageBox.question(
                self, title, f"{msg}\n\nPlay again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._start_new_game()

    # ── Speed control ──────────────────────────────────────────────────────────
    def _update_speed(self, value: int):
        delays = [2000, 1000, 600, 200, 50]
        self.ai_delay = delays[value]

    # ── Navigation ─────────────────────────────────────────────────────────────
    def _confirm_menu(self):
        self._ai_timer.stop()
        reply = QMessageBox.question(
            self, "Return to Menu", "Return to main menu? Current game will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.return_to_menu.emit()


# ─────────────────────────────────────────────────────────────────────────────
# Training Screen
# ─────────────────────────────────────────────────────────────────────────────

class TrainingScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")
        self._session: Optional[TrainingSession] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[TrainingWorker] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(16)

        title = QLabel("🎓 Training Dashboard")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_UI_ACCENT};")
        layout.addWidget(title)

        # Config group
        cfg = QGroupBox("Training Configuration")
        cfg.setStyleSheet(f"""
            QGroupBox {{ color: {COLOR_UI_ACCENT}; font-weight: bold; border: 1px solid {COLOR_UI_ACCENT};
                         border-radius: 6px; margin-top: 6px; background: {COLOR_UI_PANEL}; padding: 10px; }}
            QGroupBox::title {{ subcontrol-origin: margin; padding: 0 4px; }}
        """)
        cfg_form = QFormLayout(cfg)
        lbl_style = f"color: {COLOR_UI_TEXT}; font-size: 11px;"

        self._agent0_combo = QComboBox()
        self._agent0_combo.addItems(["Heuristic", "Random", "RL (DQN)"])
        self._agent0_combo.setStyleSheet(f"background: #333344; color: {COLOR_UI_TEXT};")

        self._agent1_combo = QComboBox()
        self._agent1_combo.addItems(["Random", "Heuristic", "RL (DQN)"])
        self._agent1_combo.setStyleSheet(f"background: #333344; color: {COLOR_UI_TEXT};")

        self._episodes_spin = QSpinBox()
        self._episodes_spin.setRange(10, 100000)
        self._episodes_spin.setValue(DEFAULT_TRAIN_EPISODES)
        self._episodes_spin.setStyleSheet(f"background: #333344; color: {COLOR_UI_TEXT};")

        for lbl_text, widget in [
            ("Agent 1 (Player 0):", self._agent0_combo),
            ("Agent 2 (Player 1):", self._agent1_combo),
            ("Episodes:", self._episodes_spin),
        ]:
            l = QLabel(lbl_text)
            l.setStyleSheet(lbl_style)
            cfg_form.addRow(l, widget)

        layout.addWidget(cfg)

        # Progress
        prog_group = QGroupBox("Progress")
        prog_group.setStyleSheet(cfg.styleSheet())
        prog_layout = QVBoxLayout(prog_group)
        self._progress_label = QLabel("Not started")
        self._progress_label.setStyleSheet(f"color: {COLOR_UI_TEXT}; font-size: 11px;")
        prog_layout.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setStyleSheet(f"""
            QProgressBar {{ background: #222233; border: 1px solid {COLOR_UI_ACCENT};
                            border-radius: 4px; text-align: center; color: {COLOR_UI_TEXT}; }}
            QProgressBar::chunk {{ background: {COLOR_UI_ACCENT}; border-radius: 4px; }}
        """)
        prog_layout.addWidget(self._progress_bar)
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(f"color: #aaaacc; font-size: 10px;")
        self._stats_label.setWordWrap(True)
        prog_layout.addWidget(self._stats_label)
        layout.addWidget(prog_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_start = make_btn("▶  Start Training", "#2a5a2a")
        self._btn_start.clicked.connect(self._start_training)
        self._btn_stop = make_btn("■  Stop", "#5a2a2a")
        self._btn_stop.clicked.connect(self._stop_training)
        self._btn_stop.setEnabled(False)
        self._btn_save = make_btn("💾 Save Stats", "#2a2a5a")
        self._btn_save.clicked.connect(self._save_stats)
        self._btn_plot = make_btn("📈 Plot Results", "#3a2a5a")
        self._btn_plot.clicked.connect(self._plot_results)
        self._btn_menu = make_btn("← Menu")
        self._btn_menu.clicked.connect(self.return_to_menu.emit)

        for btn in [self._btn_start, self._btn_stop, self._btn_save,
                    self._btn_plot, self._btn_menu]:
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    def _start_training(self):
        n = self._episodes_spin.value()
        a0 = self._make_agent(self._agent0_combo.currentText(), 0)
        a1 = self._make_agent(self._agent1_combo.currentText(), 1)

        self._session = TrainingSession(a0, a1, n_episodes=n)
        self._progress_bar.setMaximum(n)
        self._progress_bar.setValue(0)

        self._thread = QThread()
        self._worker = TrainingWorker(self._session)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress_label.setText("Training…")

    def _make_agent(self, atype: str, pid: int) -> Agent:
        if atype == "Random":
            return RandomAgent(pid)
        elif atype == "RL (DQN)":
            return RLAgent(pid)
        return HeuristicAgent(pid)

    def _stop_training(self):
        if self._worker:
            self._worker.stop()
        self._btn_stop.setEnabled(False)

    def _on_progress(self, ep: int, summary: dict):
        self._progress_bar.setValue(ep)
        self._progress_label.setText(f"Episode {ep} / {self._episodes_spin.value()}")
        wins = summary.get('wins', [0, 0])
        win_pct = summary.get('win_pct', [0, 0])
        self._stats_label.setText(
            f"Wins: {wins[0]} ({win_pct[0]}%) vs {wins[1]} ({win_pct[1]}%)  |  "
            f"Avg turns: {summary.get('avg_turns', '?')}"
        )

    def _on_finished(self, summary: dict):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress_label.setText("Training complete!")
        wins = summary.get('wins', [0, 0])
        win_pct = summary.get('win_pct', [0, 0])
        self._stats_label.setText(
            f"Final – Wins: {wins[0]} ({win_pct[0]}%) vs {wins[1]} ({win_pct[1]}%)  |  "
            f"Avg turns: {summary.get('avg_turns', '?')}"
        )
        if self._thread:
            self._thread.quit()
            self._thread.wait()

    def _save_stats(self):
        if not self._session:
            QMessageBox.warning(self, "No Data", "Run a training session first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Stats", "training_stats.json",
                                               "JSON (*.json)")
        if path:
            self._session.save_stats(path)
            QMessageBox.information(self, "Saved", f"Stats saved to:\n{path}")

    def _plot_results(self):
        if not self._session:
            QMessageBox.warning(self, "No Data", "Run a training session first.")
            return
        self._session.plot_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics Viewer Screen
# ─────────────────────────────────────────────────────────────────────────────

class StatsViewScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(14)

        title = QLabel("📊 Statistics Viewer")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_UI_ACCENT};")
        layout.addWidget(title)

        self._text = QLabel("Load a training_stats.json file to view statistics.")
        self._text.setStyleSheet(f"color: {COLOR_UI_TEXT}; font-size: 11px;")
        self._text.setWordWrap(True)
        layout.addWidget(self._text)

        btn_row = QHBoxLayout()
        btn_load = make_btn("📂 Load Stats File")
        btn_load.clicked.connect(self._load_stats)
        btn_menu = make_btn("← Menu")
        btn_menu.clicked.connect(self.return_to_menu.emit)
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_menu)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()

    def _load_stats(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Stats", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            summary = data.get('summary', {})
            lines = [f"<b>Training Summary</b><br>"]
            for k, v in summary.items():
                lines.append(f"<b>{k}:</b> {v}")
            self._text.setText("<br>".join(lines))
            self._text.setTextFormat(Qt.TextFormat.RichText)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode Selection Dialog
# ─────────────────────────────────────────────────────────────────────────────

class ModeDialog(QDialog):
    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Configure: {mode}")
        self.setStyleSheet(f"background: {COLOR_UI_BG}; color: {COLOR_UI_TEXT};")
        self.setMinimumWidth(320)

        layout = QFormLayout(self)
        layout.setSpacing(10)

        lbl_style = f"color: {COLOR_UI_TEXT}; font-size: 11px;"
        combo_style = f"background: #333344; color: {COLOR_UI_TEXT}; padding: 4px;"

        if mode == "Human vs AI":
            self.ai_combo = QComboBox()
            self.ai_combo.addItems(["Heuristic", "Random", "RL (DQN)"])
            self.ai_combo.setStyleSheet(combo_style)
            l = QLabel("AI Opponent:")
            l.setStyleSheet(lbl_style)
            layout.addRow(l, self.ai_combo)
        else:
            self.ai0_combo = QComboBox()
            self.ai0_combo.addItems(["Heuristic", "Random", "RL (DQN)"])
            self.ai0_combo.setStyleSheet(combo_style)
            self.ai1_combo = QComboBox()
            self.ai1_combo.addItems(["Random", "Heuristic", "RL (DQN)"])
            self.ai1_combo.setStyleSheet(combo_style)
            l0 = QLabel("AI Player 1:")
            l0.setStyleSheet(lbl_style)
            l1 = QLabel("AI Player 2:")
            l1.setStyleSheet(lbl_style)
            layout.addRow(l0, self.ai0_combo)
            layout.addRow(l1, self.ai1_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.setStyleSheet(f"color: {COLOR_UI_TEXT};")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Screens
        self._menu = MenuScreen()
        self._game = GameScreen()
        self._training = TrainingScreen()
        self._stats_view = StatsViewScreen()

        self._stack.addWidget(self._menu)       # 0
        self._stack.addWidget(self._game)       # 1
        self._stack.addWidget(self._training)   # 2
        self._stack.addWidget(self._stats_view) # 3

        # Connect menu signals
        self._menu.human_vs_ai.connect(self._launch_human_vs_ai)
        self._menu.ai_vs_ai.connect(self._launch_ai_vs_ai)
        self._menu.start_train.connect(lambda: self._stack.setCurrentIndex(2))
        self._menu.view_stats.connect(lambda: self._stack.setCurrentIndex(3))
        self._menu.exit_app.connect(QApplication.quit)

        # Connect return signals
        self._game.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._training.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))
        self._stats_view.return_to_menu.connect(lambda: self._stack.setCurrentIndex(0))

        self._stack.setCurrentIndex(0)

    def _launch_human_vs_ai(self):
        dlg = ModeDialog("Human vs AI", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            ai_type = dlg.ai_combo.currentText()
            self._game.start_human_vs_ai(ai_type)
            self._stack.setCurrentIndex(1)

    def _launch_ai_vs_ai(self):
        dlg = ModeDialog("AI vs AI", self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            a0 = dlg.ai0_combo.currentText()
            a1 = dlg.ai1_combo.currentText()
            self._game.start_ai_vs_ai(a0, a1, delay=AI_FAST_DELAY_MS)
            self._stack.setCurrentIndex(1)
