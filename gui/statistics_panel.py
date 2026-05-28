"""
StatisticsPanel – side panel displaying game info, AI metrics, and training data.
"""

from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QFrame, QScrollArea, QSizePolicy,
    QPushButton, QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from utils.config import (
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def _lbl(text: str, bold=False, size=10, color=COLOR_UI_TEXT) -> QLabel:
    l = QLabel(text)
    f = QFont("Arial", size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def _group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{
            color: {COLOR_UI_ACCENT};
            font-weight: bold;
            font-size: 10px;
            border: 1px solid {COLOR_UI_ACCENT};
            border-radius: 5px;
            margin-top: 6px;
            background: {COLOR_UI_PANEL};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
        }}
    """)
    return g


class StatRow(QWidget):
    """A single key-value stat row."""

    def __init__(self, key: str, value: str = "—"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        self._key = _lbl(key, size=9, color="#aaaacc")
        self._val = _lbl(value, bold=True, size=9, color=COLOR_UI_TEXT)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._key)
        layout.addStretch()
        layout.addWidget(self._val)
        self.setStyleSheet("background: transparent;")

    def update_value(self, value: str):
        self._val.setText(value)


class StatisticsPanel(QWidget):
    """
    Right-hand side statistics panel.
    Shows current game state info, AI decision data, and training metrics.
    """

    end_turn_requested = pyqtSignal()
    undo_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(270)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")

        self._stat_rows: Dict[str, StatRow] = {}
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        container = QWidget()
        container.setStyleSheet(f"background: {COLOR_UI_BG};")
        self._main = QVBoxLayout(container)
        self._main.setContentsMargins(8, 8, 8, 8)
        self._main.setSpacing(8)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Turn info ──────────────────────────────────────────────────────────
        self._turn_label = _lbl("Turn 1", bold=True, size=13, color=COLOR_UI_ACCENT)
        self._turn_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._turn_label)

        self._player_label = _lbl("YOUR TURN", bold=True, size=11, color="#66ff99")
        self._player_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._player_label)

        self._phase_label = _lbl("Phase: PLAY", size=9, color="#aaaaff")
        self._phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._phase_label)

        # ── Game stats ─────────────────────────────────────────────────────────
        g_game = _group("Game State")
        gl = QVBoxLayout(g_game)
        gl.setSpacing(2)
        for key in ['Your Stockpile', 'AI Stockpile', 'Deck Remaining', 'Completed Seqs']:
            row = StatRow(key)
            gl.addWidget(row)
            self._stat_rows[key] = row
        self._main.addWidget(g_game)

        # ── Session stats ──────────────────────────────────────────────────────
        g_session = _group("Session")
        sl = QVBoxLayout(g_session)
        sl.setSpacing(2)
        for key in ['Games Played', 'Human Wins', 'AI Wins', 'Win Rate', 'Avg Turns']:
            row = StatRow(key)
            sl.addWidget(row)
            self._stat_rows[key] = row
        self._main.addWidget(g_session)

        # ── AI info ────────────────────────────────────────────────────────────
        g_ai = _group("AI Info")
        al = QVBoxLayout(g_ai)
        al.setSpacing(2)
        self._ai_action_label = _lbl("Waiting...", size=9, color="#ddddff")
        self._ai_action_label.setWordWrap(True)
        al.addWidget(self._ai_action_label)

        for key in ['AI Type', 'Epsilon', 'Avg Loss', 'Episodes']:
            row = StatRow(key)
            al.addWidget(row)
            self._stat_rows[key] = row
        self._main.addWidget(g_ai)

        # ── Controls ───────────────────────────────────────────────────────────
        g_ctrl = _group("Controls")
        cl = QVBoxLayout(g_ctrl)
        cl.setSpacing(6)

        self._btn_end_turn = self._make_btn("⏭  End Turn / Discard", "#336633")
        self._btn_end_turn.clicked.connect(self.end_turn_requested.emit)
        cl.addWidget(self._btn_end_turn)

        self._main.addWidget(g_ctrl)
        self._main.addStretch()

        # ── Mini chart (if matplotlib available) ──────────────────────────────
        if _MPL:
            self._chart_widget = MiniChart()
            self._main.addWidget(self._chart_widget)

    def _make_btn(self, text: str, bg: str = COLOR_BUTTON) -> QPushButton:
        btn = QPushButton(text)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {COLOR_UI_TEXT};
                border: none;
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {COLOR_BUTTON_HOVER};
            }}
            QPushButton:disabled {{
                background: #333344;
                color: #666677;
            }}
        """)
        return btn

    # ── Public update methods ─────────────────────────────────────────────────
    def update_game_state(self, state, human_idx: int = 0):
        """Refresh all game-state-driven stats."""
        from game.game_state import GameState
        player = state.players[human_idx]
        ai_idx = 1 - human_idx
        ai = state.players[ai_idx]

        self._turn_label.setText(f"Turn {state.turn_number}")

        is_human_turn = state.current_player_idx == human_idx
        self._player_label.setText(
            "YOUR TURN" if is_human_turn else f"⏳ {ai.name}'s Turn"
        )
        self._player_label.setStyleSheet(
            f"color: {'#66ff99' if is_human_turn else '#ffaa44'}; "
            f"font-weight: bold; font-size: 11px; background: transparent;"
        )
        self._phase_label.setText(f"Phase: {state.phase.upper()}")

        self._stat_rows['Your Stockpile'].update_value(str(player.stockpile_size))
        self._stat_rows['AI Stockpile'].update_value(str(ai.stockpile_size))
        self._stat_rows['Deck Remaining'].update_value(str(state.deck.remaining))
        self._stat_rows['Completed Seqs'].update_value(str(state.completed_sequences))

        self._btn_end_turn.setEnabled(is_human_turn)

    def update_session_stats(self, games: int, wins: List[int], avg_turns: float,
                              human_idx: int = 0):
        self._stat_rows['Games Played'].update_value(str(games))
        self._stat_rows['Human Wins'].update_value(str(wins[human_idx]))
        self._stat_rows['AI Wins'].update_value(str(wins[1 - human_idx]))
        win_rate = (wins[human_idx] / games * 100) if games > 0 else 0.0
        self._stat_rows['Win Rate'].update_value(f"{win_rate:.1f}%")
        self._stat_rows['Avg Turns'].update_value(f"{avg_turns:.1f}")

    def update_ai_info(self, agent, last_action=None):
        atype = type(agent).__name__.replace('Agent', '')
        self._stat_rows['AI Type'].update_value(atype)

        if last_action:
            self._ai_action_label.setText(f"Last: {last_action}")

        if hasattr(agent, 'get_stats'):
            stats = agent.get_stats()
            self._stat_rows['Epsilon'].update_value(f"{stats.get('epsilon', '—')}")
            loss = stats.get('avg_loss', 0)
            self._stat_rows['Avg Loss'].update_value(f"{loss:.5f}" if loss else "—")
            self._stat_rows['Episodes'].update_value(str(stats.get('steps', '—')))
        else:
            self._stat_rows['Epsilon'].update_value("—")
            self._stat_rows['Avg Loss'].update_value("—")
            self._stat_rows['Episodes'].update_value("—")

    def update_win_chart(self, episodes: List[int], win_rates: List[float]):
        if _MPL and hasattr(self, '_chart_widget'):
            self._chart_widget.update_data(episodes, win_rates)


# ─────────────────────────────────────────────────────────────────────────────
# Mini embedded matplotlib chart
# ─────────────────────────────────────────────────────────────────────────────

if _MPL:
    class MiniChart(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._fig = Figure(figsize=(2.2, 1.5), facecolor='#1e1e2e')
            self._ax = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(140)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel("Win Rate Trend")
            lbl.setStyleSheet(f"color: {COLOR_UI_ACCENT}; font-size: 9px; "
                              "font-weight: bold; background: transparent;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            layout.addWidget(self._canvas)
            self._draw_empty()

        def _draw_empty(self):
            self._ax.cla()
            self._ax.set_facecolor('#1e1e2e')
            self._ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                          color='#666677', fontsize=8,
                          transform=self._ax.transAxes)
            self._ax.tick_params(colors='#555566', labelsize=7)
            for spine in self._ax.spines.values():
                spine.set_color('#333344')
            self._fig.tight_layout(pad=0.3)
            self._canvas.draw()

        def update_data(self, episodes: List[int], win_rates: List[float]):
            if not episodes:
                return
            self._ax.cla()
            self._ax.set_facecolor('#1e1e2e')
            self._ax.plot(episodes, win_rates, color='#7c6af7', linewidth=1.5)
            self._ax.axhline(0.5, color='#555566', linestyle='--', linewidth=0.8)
            self._ax.set_ylim(0, 1)
            self._ax.tick_params(colors='#888899', labelsize=6)
            for spine in self._ax.spines.values():
                spine.set_color('#333344')
            self._ax.set_ylabel('Win', color='#888899', fontsize=7)
            self._fig.tight_layout(pad=0.3)
            self._canvas.draw()
