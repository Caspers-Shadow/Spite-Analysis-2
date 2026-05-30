"""
StatisticsPanel – right-hand side panel showing live game info,
AI decision data, session metrics, and a rolling win-rate mini chart.
"""

from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QPushButton, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from utils.config import (
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False


def _lbl(text, bold=False, size=10, color=COLOR_UI_TEXT):
    l = QLabel(text)
    f = QFont("Arial", size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{
            color: {COLOR_UI_ACCENT}; font-weight: bold; font-size: 10px;
            border: 1px solid {COLOR_UI_ACCENT}; border-radius: 5px;
            margin-top: 6px; background: {COLOR_UI_PANEL};
        }}
        QGroupBox::title {{ subcontrol-origin: margin; padding: 0 4px; }}
    """)
    return g


class StatRow(QWidget):
    def __init__(self, key: str, value: str = "—"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        self._key = _lbl(key, size=9, color="#aaaacc")
        self._val = _lbl(value, bold=True, size=9)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._key)
        layout.addStretch()
        layout.addWidget(self._val)
        self.setStyleSheet("background: transparent;")

    def set(self, value: str):
        self._val.setText(value)


class StatisticsPanel(QWidget):
    end_turn_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(210)
        self.setMaximumWidth(255)
        self.setStyleSheet(f"background: {COLOR_UI_BG};")
        self._rows: Dict[str, StatRow] = {}
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

        # Turn / phase header
        self._turn_lbl = _lbl("Turn 1", bold=True, size=13, color=COLOR_UI_ACCENT)
        self._turn_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._turn_lbl)

        self._player_lbl = _lbl("YOUR TURN", bold=True, size=11, color="#66ff99")
        self._player_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._player_lbl)

        self._phase_lbl = _lbl("Phase: PLAY", size=9, color="#aaaaff")
        self._phase_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main.addWidget(self._phase_lbl)

        # ── Game state ─────────────────────────────────────────────────────────
        g = _group("Game State")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)
        for k in ["Your Stockpile", "AI Stockpile", "Deck Remaining",
                  "Completed Seqs", "Hand Refills", "Unlock Status"]:
            r = StatRow(k)
            gl.addWidget(r)
            self._rows[k] = r
        self._main.addWidget(g)

        # ── Session ────────────────────────────────────────────────────────────
        g2 = _group("This Session")
        g2l = QVBoxLayout(g2)
        g2l.setSpacing(2)
        for k in ["Games Played", "Your Wins", "AI Wins", "Win Rate", "Avg Turns"]:
            r = StatRow(k)
            g2l.addWidget(r)
            self._rows[k] = r
        self._main.addWidget(g2)

        # ── AI info ────────────────────────────────────────────────────────────
        g3 = _group("AI")
        g3l = QVBoxLayout(g3)
        g3l.setSpacing(2)
        self._ai_action_lbl = _lbl("—", size=8, color="#ddddff")
        self._ai_action_lbl.setWordWrap(True)
        g3l.addWidget(self._ai_action_lbl)
        for k in ["AI Type", "Epsilon", "Avg Loss"]:
            r = StatRow(k)
            g3l.addWidget(r)
            self._rows[k] = r
        self._main.addWidget(g3)

        self._main.addStretch()

        # ── Embedded mini win-rate chart ───────────────────────────────────────
        if _MPL:
            self._mini_chart = _MiniWinChart()
            self._main.addWidget(self._mini_chart)

    # ── Public update API ──────────────────────────────────────────────────────
    def update_game_state(self, state, human_idx: int = 0):
        from game.game_state import GameState
        player = state.players[human_idx]
        ai     = state.players[1 - human_idx]

        self._turn_lbl.setText(f"Turn {state.turn_number}")
        is_human = state.current_player_idx == human_idx
        self._player_lbl.setText("YOUR TURN" if is_human else f"⏳ {ai.name}…")
        self._player_lbl.setStyleSheet(
            f"color: {'#66ff99' if is_human else '#ffaa44'}; "
            "font-weight: bold; font-size: 11px; background: transparent;")
        self._phase_lbl.setText(f"Phase: {state.phase.upper()}")

        self._rows["Your Stockpile"].set(str(player.stockpile_size))
        self._rows["AI Stockpile"].set(str(ai.stockpile_size))
        self._rows["Deck Remaining"].set(str(state.deck.remaining))
        self._rows["Completed Seqs"].set(str(state.completed_sequences))
        self._rows["Hand Refills"].set(str(state.hand_refills))
        # Unlock status
        hu = "✓ Unlocked" if state.player_unlocked[human_idx] else f"✗ Need Ace ({state.pile_starts_by[1-human_idx]}/4 opp)"
        self._rows["Unlock Status"].set(hu)

    def update_session(self, games: int, wins: List[int],
                       avg_turns: float, human_idx: int = 0):
        self._rows["Games Played"].set(str(games))
        self._rows["Your Wins"].set(str(wins[human_idx]))
        self._rows["AI Wins"].set(str(wins[1 - human_idx]))
        pct = wins[human_idx] / games * 100 if games else 0
        self._rows["Win Rate"].set(f"{pct:.1f}%")
        self._rows["Avg Turns"].set(f"{avg_turns:.1f}")

    def update_ai_info(self, agent, last_action=None):
        self._rows["AI Type"].set(type(agent).__name__.replace("Agent", ""))
        if last_action:
            self._ai_action_lbl.setText(str(last_action)[:60])
        if hasattr(agent, "get_stats"):
            s = agent.get_stats()
            self._rows["Epsilon"].set(str(s.get("epsilon", "—")))
            loss = s.get("avg_loss", 0)
            self._rows["Avg Loss"].set(f"{loss:.5f}" if loss else "—")
        else:
            self._rows["Epsilon"].set("N/A")
            self._rows["Avg Loss"].set("N/A")

    def push_game_result(self, human_won: bool):
        """Feed a single game result into the mini win-rate chart."""
        if _MPL and hasattr(self, "_mini_chart"):
            self._mini_chart.push(human_won)


# ─────────────────────────────────────────────────────────────────────────────
# Mini rolling win-rate chart (embedded in panel)
# ─────────────────────────────────────────────────────────────────────────────

if _MPL:
    class _MiniWinChart(QWidget):
        """Tiny rolling win-rate chart that updates after every game."""

        WINDOW = 15   # rolling window size

        def __init__(self, parent=None):
            super().__init__(parent)
            self._history: List[int] = []   # 1 = human win, 0 = loss
            self._fig = Figure(figsize=(2.1, 1.4), facecolor="#1e1e2e")
            self._ax  = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(130)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel("Session Win Rate")
            lbl.setStyleSheet(f"color: {COLOR_UI_ACCENT}; font-size: 9px; "
                              "font-weight: bold; background: transparent;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            layout.addWidget(self._canvas)
            self._draw()

        def push(self, won: bool):
            self._history.append(1 if won else 0)
            self._draw()

        def _draw(self):
            self._ax.cla()
            ax = self._ax
            ax.set_facecolor("#1e1e2e")
            for spine in ax.spines.values():
                spine.set_color("#333344")
            ax.tick_params(colors="#555566", labelsize=6)

            if len(self._history) < 2:
                ax.text(0.5, 0.5, "Waiting for games…",
                        ha="center", va="center", color="#555566",
                        fontsize=7, transform=ax.transAxes)
                self._canvas.draw()
                return

            xs, ys = [], []
            for i in range(1, len(self._history) + 1):
                chunk = self._history[max(0, i - self.WINDOW): i]
                xs.append(i)
                ys.append(sum(chunk) / len(chunk))

            ax.plot(xs, ys, color="#7c6af7", linewidth=1.5)
            ax.axhline(0.5, color="#555566", linestyle="--", linewidth=0.8)
            ax.fill_between(xs, ys, 0.5,
                            where=[y > 0.5 for y in ys], alpha=0.2, color="#50c878")
            ax.fill_between(xs, ys, 0.5,
                            where=[y <= 0.5 for y in ys], alpha=0.2, color="#ff7f7f")
            ax.set_ylim(0, 1)
            ax.set_xlim(1, max(xs))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
            self._fig.tight_layout(pad=0.3)
            self._canvas.draw()