"""
StatsViewScreen – full statistics dashboard with embedded matplotlib graphs.

Shows per-matchup history across all AI/Human combinations with:
  • Win-rate trend chart (rolling window)
  • Game-length distribution histogram
  • All-matchups comparison bar chart
  • Numeric summary table
"""

import time
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QScrollArea, QFrame,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from utils.stats_manager import StatsManager
from utils.config import (
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
)

try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ── palette used for all charts ───────────────────────────────────────────────
BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
TEXT    = "#e0e0f0"
ACCENT  = "#7c6af7"
GREEN   = "#50c878"
SALMON  = "#ff7f7f"
GREY    = "#888899"
YELLOW  = "#ffe066"

MATCHUP_COLORS = [
    "#7c6af7", "#50c878", "#ff7f7f", "#ffe066",
    "#56b4e9", "#e69f00", "#cc79a7", "#009e73",
]


def _lbl(text, bold=False, size=10, color=TEXT):
    l = QLabel(text)
    f = QFont("Arial", size)
    f.setBold(bold)
    l.setFont(f)
    l.setStyleSheet(f"color: {color}; background: transparent;")
    return l


def _btn(text, bg=COLOR_BUTTON):
    b = QPushButton(text)
    b.setStyleSheet(f"""
        QPushButton {{
            background:{bg}; color:{TEXT}; border:none;
            border-radius:6px; padding:6px 14px;
            font-size:11px; font-weight:bold;
        }}
        QPushButton:hover {{ background:{COLOR_BUTTON_HOVER}; }}
    """)
    return b


def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{
            color:{ACCENT}; font-weight:bold; font-size:10px;
            border:1px solid {ACCENT}; border-radius:5px;
            margin-top:6px; background:{PANEL};
        }}
        QGroupBox::title {{ subcontrol-origin:margin; padding:0 4px; }}
    """)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Embedded matplotlib canvas
# ─────────────────────────────────────────────────────────────────────────────

class ChartCanvas(QWidget):
    """Wraps a matplotlib Figure in a QWidget."""

    def __init__(self, figsize=(8, 5), parent=None):
        super().__init__(parent)
        if not _MPL:
            layout = QVBoxLayout(self)
            layout.addWidget(_lbl("matplotlib not installed.\npip install matplotlib",
                                  color=SALMON, size=12))
            return
        self._fig = Figure(figsize=figsize, facecolor=BG, tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def figure(self):
        return self._fig if _MPL else None

    def draw(self):
        if _MPL:
            self._canvas.draw()

    def clear(self):
        if _MPL:
            self._fig.clf()


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GREY, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333344")
    ax.set_title(title, color=TEXT, fontsize=10, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=GREY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=GREY, fontsize=8)
    ax.grid(True, alpha=0.15, color=GREY)


# ─────────────────────────────────────────────────────────────────────────────
# Main Stats Screen
# ─────────────────────────────────────────────────────────────────────────────

class StatsViewScreen(QWidget):
    """Full statistics dashboard."""

    return_to_menu = pyqtSignal()

    def __init__(self, stats_manager: Optional[StatsManager] = None, parent=None):
        super().__init__(parent)
        self.mgr = stats_manager or StatsManager()
        self.setStyleSheet(f"background:{BG};")
        self._selected_matchup: Optional[str] = None
        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # Title bar
        title_row = QHBoxLayout()
        title = _lbl("📊  Statistics Dashboard", bold=True, size=18, color=ACCENT)
        title_row.addWidget(title)
        title_row.addStretch()
        btn_refresh = _btn("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_clear = _btn("🗑  Clear All", "#5a1a1a")
        btn_clear.clicked.connect(self._clear_stats)
        btn_menu = _btn("← Menu")
        btn_menu.clicked.connect(self.return_to_menu.emit)
        for b in [btn_refresh, btn_clear, btn_menu]:
            title_row.addWidget(b)
        root.addLayout(title_row)

        # Splitter: left list + right charts
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle {{ background:{ACCENT}; }}")

        # ── Left: matchup list + summary table ────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background:{PANEL}; border-radius:8px;")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)

        left_lay.addWidget(_lbl("Matchups", bold=True, size=11, color=ACCENT))

        self._list = QListWidget()
        self._list.setStyleSheet(f"""
            QListWidget {{
                background:{BG}; color:{TEXT}; border:1px solid #333344;
                border-radius:4px; font-size:10px;
            }}
            QListWidget::item:selected {{ background:{ACCENT}; color:#fff; }}
            QListWidget::item:hover    {{ background:#333355; }}
        """)
        self._list.currentItemChanged.connect(self._on_matchup_selected)
        left_lay.addWidget(self._list, stretch=2)

        # Summary table for selected matchup
        sum_grp = _group("Summary")
        self._sum_layout = QVBoxLayout(sum_grp)
        self._sum_layout.setSpacing(2)
        self._sum_rows: dict = {}
        for row_key in ["Games", "P0 Wins", "P1 Wins", "P0 Win %", "P1 Win %", "Avg Turns"]:
            row = QHBoxLayout()
            k = _lbl(row_key + ":", size=9, color=GREY)
            v = _lbl("—", bold=True, size=9)
            row.addWidget(k)
            row.addStretch()
            row.addWidget(v)
            self._sum_layout.addLayout(row)
            self._sum_rows[row_key] = v
        left_lay.addWidget(sum_grp, stretch=1)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

        # ── Right: charts ──────────────────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet(f"background:{BG};")
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 0, 0, 0)
        right_lay.setSpacing(6)

        # Tab row
        tab_row = QHBoxLayout()
        self._tab_btns = []
        for label in ["Win Rate Trend", "Game Length", "All Matchups"]:
            tb = _btn(label, bg="#2a2a4a")
            tb.clicked.connect(self._make_tab_handler(label))
            tab_row.addWidget(tb)
            self._tab_btns.append((label, tb))
        tab_row.addStretch()
        right_lay.addLayout(tab_row)

        self._chart = ChartCanvas(figsize=(8, 5))
        right_lay.addWidget(self._chart, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 740])

        root.addWidget(splitter, stretch=1)

        self._current_tab = "Win Rate Trend"
        self.refresh()

    # ── Refresh ────────────────────────────────────────────────────────────────
    def refresh(self):
        """Reload matchup list and redraw active chart."""
        self._list.clear()
        matchups = self.mgr.matchups()
        if not matchups:
            self._list.addItem("No games recorded yet")
        for key in matchups:
            s = self.mgr.summary(key)
            item = QListWidgetItem(f"{key}  ({s['games']} games)")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._list.addItem(item)

        # Re-select previously selected matchup if still present
        if self._selected_matchup and self._selected_matchup in matchups:
            for i in range(self._list.count()):
                if self._list.item(i).data(Qt.ItemDataRole.UserRole) == self._selected_matchup:
                    self._list.setCurrentRow(i)
                    break
        elif matchups:
            self._list.setCurrentRow(0)
        else:
            self._draw_empty("Play some games to see statistics here!")

    def _on_matchup_selected(self, current, _previous):
        if current is None:
            return
        key = current.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        self._selected_matchup = key
        self._update_summary(key)
        self._draw_tab(self._current_tab)

    def _update_summary(self, key: str):
        s = self.mgr.summary(key)
        self._sum_rows["Games"].setText(str(s.get("games", 0)))
        self._sum_rows["P0 Wins"].setText(str(s.get("wins_p0", 0)))
        self._sum_rows["P1 Wins"].setText(str(s.get("wins_p1", 0)))
        self._sum_rows["P0 Win %"].setText(f"{s.get('win_pct_p0', 0)}%")
        self._sum_rows["P1 Win %"].setText(f"{s.get('win_pct_p1', 0)}%")
        self._sum_rows["Avg Turns"].setText(str(s.get("avg_turns", "—")))

    # ── Tab switching ──────────────────────────────────────────────────────────
    def _make_tab_handler(self, label):
        def handler():
            self._current_tab = label
            for lbl, btn in self._tab_btns:
                btn.setStyleSheet(btn.styleSheet().replace(
                    "#7c6af7" if lbl == label else "#2a2a4a",
                    "#7c6af7" if lbl == label else "#2a2a4a",
                ))
                bg = ACCENT if lbl == label else "#2a2a4a"
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:{bg}; color:{TEXT}; border:none;
                        border-radius:6px; padding:6px 14px;
                        font-size:11px; font-weight:bold;
                    }}
                    QPushButton:hover {{ background:{COLOR_BUTTON_HOVER}; }}
                """)
            self._draw_tab(label)
        return handler

    def _draw_tab(self, tab: str):
        if tab == "Win Rate Trend":
            self._draw_win_rate()
        elif tab == "Game Length":
            self._draw_game_length()
        elif tab == "All Matchups":
            self._draw_all_matchups()

    # ── Chart renderers ────────────────────────────────────────────────────────
    def _draw_empty(self, msg="No data"):
        if not _MPL:
            return
        self._chart.clear()
        fig = self._chart.figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                color=GREY, fontsize=13, transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self._chart.draw()

    def _draw_win_rate(self):
        if not _MPL:
            return
        key = self._selected_matchup
        if not key:
            self._draw_empty("Select a matchup from the list")
            return

        xs, ys = self.mgr.rolling_win_rate(key, window=20)
        if not xs:
            self._draw_empty(f"Not enough games yet for '{key}'")
            return

        parts = key.split(" vs ")
        p0 = parts[0] if parts else "P0"
        p1 = parts[1] if len(parts) > 1 else "P1"

        self._chart.clear()
        fig = self._chart.figure()
        ax = fig.add_subplot(111)
        _style_ax(ax, title=f"Win Rate Trend – {key}",
                  xlabel="Game #", ylabel=f"{p0} Win Rate")

        ax.plot(xs, ys, color=ACCENT, linewidth=2, label=f"{p0} win rate (rolling 20)")
        ax.axhline(0.5, color=GREY, linestyle="--", linewidth=1, alpha=0.6, label="50 %")
        ax.fill_between(xs, ys, 0.5, where=[y > 0.5 for y in ys],
                         alpha=0.15, color=GREEN, label=f"{p0} ahead")
        ax.fill_between(xs, ys, 0.5, where=[y <= 0.5 for y in ys],
                         alpha=0.15, color=SALMON, label=f"{p1} ahead")
        ax.set_ylim(0, 1)
        ax.set_xlim(1, max(xs))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        legend = ax.legend(fontsize=8, facecolor=PANEL, edgecolor="#444",
                           labelcolor=TEXT)
        self._chart.draw()

    def _draw_game_length(self):
        if not _MPL:
            return
        key = self._selected_matchup
        if not key:
            self._draw_empty("Select a matchup from the list")
            return

        games = self.mgr.games(key)
        if not games:
            self._draw_empty(f"No games for '{key}'")
            return

        turns = [g["turns"] for g in games]
        p0_turns = [g["turns"] for g in games if g["winner"] == 0]
        p1_turns = [g["turns"] for g in games if g["winner"] == 1]

        parts = key.split(" vs ")
        p0, p1 = (parts[0], parts[1]) if len(parts) > 1 else ("P0", "P1")

        self._chart.clear()
        fig = self._chart.figure()
        ax = fig.add_subplot(111)
        _style_ax(ax, title=f"Game Length Distribution – {key}",
                  xlabel="Turns", ylabel="Games")

        bins = range(min(turns), max(turns) + 5, 4)
        if p0_turns:
            ax.hist(p0_turns, bins=bins, alpha=0.7, color=ACCENT,
                    label=f"{p0} wins (avg {sum(p0_turns)/len(p0_turns):.0f}t)")
        if p1_turns:
            ax.hist(p1_turns, bins=bins, alpha=0.7, color=SALMON,
                    label=f"{p1} wins (avg {sum(p1_turns)/len(p1_turns):.0f}t)")

        avg = sum(turns) / len(turns)
        ax.axvline(avg, color=YELLOW, linestyle="--", linewidth=1.5,
                   label=f"Overall avg {avg:.0f}t")
        ax.legend(fontsize=8, facecolor=PANEL, edgecolor="#444", labelcolor=TEXT)
        self._chart.draw()

    def _draw_all_matchups(self):
        if not _MPL:
            return
        all_m = self.mgr.all_summaries()
        if not all_m:
            self._draw_empty("No games recorded yet — play some games first!")
            return

        keys   = list(all_m.keys())
        p0pct  = [all_m[k]["win_pct_p0"] for k in keys]
        p1pct  = [all_m[k]["win_pct_p1"] for k in keys]
        games  = [all_m[k]["games"] for k in keys]
        colors = MATCHUP_COLORS[:len(keys)]

        self._chart.clear()
        fig = self._chart.figure()

        # Left: stacked win % bar chart
        ax1 = fig.add_subplot(121)
        _style_ax(ax1, title="Win % by Matchup", xlabel="", ylabel="Win %")

        y_pos = range(len(keys))
        ax1.barh(y_pos, p0pct, color=ACCENT, alpha=0.85, label="P0 wins")
        ax1.barh(y_pos, p1pct, left=p0pct, color=SALMON, alpha=0.85, label="P1 wins")
        draw_pct = [100 - p - q for p, q in zip(p0pct, p1pct)]
        ax1.barh(y_pos, draw_pct, left=[p + q for p, q in zip(p0pct, p1pct)],
                 color=GREY, alpha=0.5, label="Draws")
        ax1.set_yticks(list(y_pos))
        ax1.set_yticklabels(
            [k.replace(" vs ", "\nvs\n") for k in keys],
            fontsize=7, color=TEXT,
        )
        ax1.set_xlim(0, 100)
        ax1.axvline(50, color=YELLOW, linestyle="--", linewidth=1, alpha=0.6)
        ax1.legend(fontsize=7, facecolor=PANEL, edgecolor="#444", labelcolor=TEXT,
                   loc="lower right")

        # Right: games played bubble chart
        ax2 = fig.add_subplot(122)
        _style_ax(ax2, title="Games Played per Matchup", xlabel="", ylabel="")

        ax2.barh(list(y_pos), games, color=colors, alpha=0.85)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(
            [k.replace(" vs ", "\nvs\n") for k in keys],
            fontsize=7, color=TEXT,
        )
        for i, (g, p) in enumerate(zip(games, p0pct)):
            ax2.text(g + 0.3, i, f"{g}", va="center", color=TEXT, fontsize=8)

        fig.tight_layout(pad=1.2)
        self._chart.draw()

    # ── Actions ────────────────────────────────────────────────────────────────
    def _clear_stats(self):
        reply = QMessageBox.question(
            self, "Clear Stats",
            "Delete ALL recorded game statistics? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.mgr.clear()
            self._selected_matchup = None
            self.refresh()
            self._draw_empty("Stats cleared.")