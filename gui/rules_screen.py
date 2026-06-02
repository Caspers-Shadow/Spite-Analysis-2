"""Rules & Help screen – shown from the main menu."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextBrowser, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from utils.config import (
    COLOR_UI_BG, COLOR_UI_PANEL, COLOR_UI_TEXT,
    COLOR_UI_ACCENT, COLOR_BUTTON, COLOR_BUTTON_HOVER,
)

BTN = f"""
    QPushButton {{
        background:{COLOR_BUTTON}; color:{COLOR_UI_TEXT}; border:none;
        border-radius:6px; padding:7px 16px; font-size:11px; font-weight:bold;
    }}
    QPushButton:hover   {{ background:{COLOR_BUTTON_HOVER}; }}
    QPushButton:checked {{ background:{COLOR_UI_ACCENT}; }}
"""

SECTIONS = {
    "Game Rules": """
<h2 style="color:#7c6af7;">🃏 Spite Analysis — Game Rules</h2>

<h3 style="color:#ffe066;">Objective</h3>
<p>Be the first player to completely empty your personal <b>Stockpile</b> by playing cards
onto shared <b>Building Piles</b> in ascending order (Ace → Queen).</p>

<h3 style="color:#ffe066;">Setup</h3>
<ul>
<li>Two decks (104 cards total) are shuffled together.</li>
<li>Each player receives a face-down stockpile of <b>13 cards</b>. The top card is flipped face-up.</li>
<li>Each player draws <b>5 cards</b> into their hand.</li>
<li>Four empty Building Pile slots sit in the centre.</li>
</ul>

<h3 style="color:#ffe066;">Turn Structure</h3>
<ol>
<li><b>Draw</b> — draw cards until you hold 5.</li>
<li><b>Play Phase</b> — play any number of legal cards from your hand, stockpile top,
or discard pile tops onto building piles.</li>
<li><b>Discard</b> — place exactly one hand card onto one of your 4 personal discard piles.
This ends your turn.</li>
</ol>

<h3 style="color:#ffe066;">🔓 Building Pile Unlock Rule</h3>
<p>You may only play <b>non-Ace</b> cards onto building piles after you are <i>unlocked</i>.
You become unlocked when:</p>
<ul>
<li><b>(a)</b> You personally play an Ace to start a building pile, <b>OR</b></li>
<li><b>(b)</b> Your opponent has cumulatively started all 4 building pile slots (placed 4+ Aces lifetime).</li>
</ul>
<p>Playing an Ace is <i>always</i> permitted — it is how you unlock yourself. The side panel shows your current unlock status.</p>

<h3 style="color:#ffe066;">Building Piles</h3>
<ul>
<li>Up to 4 shared piles in the centre.</li>
<li>Must begin with an <b>Ace</b> and ascend exactly by +1 each card.</li>
<li>Valid sequence: <b>A → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → J → Q</b></li>
<li>When a pile reaches <b>Queen</b>, it is cleared and the slot can start again.</li>
</ul>

<h3 style="color:#ffe066;">Hand Refill Rule</h3>
<p>If you play all 5 cards from your hand during your turn, you immediately draw
a fresh hand of 5 cards and your turn <b>continues</b> — you do not discard yet.</p>

<h3 style="color:#ffe066;">Wild Cards (Kings)</h3>
<ul>
<li>🔴 <b>Red King</b> — can represent <b>any value from Ace to Queen (1–12)</b>.</li>
<li>⚫ <b>Black King</b> — can represent <b>any value from 2 to Queen (2–12)</b>, but NOT Ace.</li>
<li>Once played to a building pile, a wild card permanently takes that value for that pile.</li>
<li>Wild cards glow on screen: red stripe for Red Kings, dark stripe for Black Kings.</li>
</ul>

<h3 style="color:#ffe066;">Card Sources Each Turn</h3>
<p>You may play from any of these during your play phase:</p>
<ul>
<li>Your <b>hand</b> (any card)</li>
<li>The <b>top card</b> of your stockpile</li>
<li>The <b>top card</b> of any of your 4 personal discard piles</li>
</ul>

<h3 style="color:#ffe066;">Win Condition</h3>
<p>You win the moment your <b>stockpile reaches zero cards</b>. The game ends instantly.</p>
""",

    "UI Guide": """
<h2 style="color:#7c6af7;">🖱️ How to Play — UI Controls</h2>

<h3 style="color:#ffe066;">Playing a Card</h3>
<ol>
<li><b>Click a card</b> in your hand, stockpile, or discard pile top to <i>select</i> it
(it will highlight in yellow).</li>
<li><b>Click a Building Pile</b> to play the selected card there.
Piles you can legally play on glow green.</li>
<li>If the move is invalid, nothing happens — try a different pile or card.</li>
</ol>

<h3 style="color:#ffe066;">Ending Your Turn (Discarding)</h3>
<p>You must discard exactly one hand card to end your turn. Two ways:</p>
<ul>
<li><b>Select a hand card → click one of your 4 Discard Piles</b> (they glow green when a hand card is selected).</li>
<li><b>Click the "⏭ End Turn" button</b> at the bottom of your area — it auto-selects the best card to discard.</li>
</ul>

<h3 style="color:#ffe066;">Deselecting</h3>
<p>Click the same card again to deselect it, or simply click a different card to switch selection.</p>

<h3 style="color:#ffe066;">Toolbar</h3>
<ul>
<li><b>← Menu</b> — returns to main menu (confirms if game is in progress).</li>
<li><b>New Game</b> — starts a fresh game with the same settings.</li>
<li><b>AI Speed slider</b> — controls how fast the AI takes its turns (Very Slow → Instant).</li>
</ul>

<h3 style="color:#ffe066;">Side Panel</h3>
<ul>
<li><b>Stockpile counters</b> — shows how many cards remain in each player's stockpile.</li>
<li><b>Unlock Status</b> — whether you can play non-Ace cards yet (✓ or ✗).</li>
<li><b>Deck Remaining</b> — cards left in the draw pile.</li>
<li><b>Completed Sequences</b> — how many building piles have reached Queen and been cleared.</li>
<li><b>Hand Refills</b> — how many times a hand was refilled mid-turn.</li>
<li><b>Win Rate chart</b> — rolling win rate across your current session.</li>
</ul>

<h3 style="color:#ffe066;">Status Bar</h3>
<p>The gold text at the very bottom of the board always tells you what to do next.</p>

<h3 style="color:#ffe066;">AI vs AI Mode</h3>
<ul>
<li>Use the speed slider to slow down and watch the AI play.</li>
<li>The game auto-restarts after each result.</li>
<li>Session stats update in the side panel.</li>
</ul>
""",

    "AI Guide": """
<h2 style="color:#7c6af7;">🤖 AI Agents — Guide & Comparison</h2>

<h3 style="color:#ffe066;">Random Agent</h3>
<p>Plays a completely random legal action each step. 40 % of the time it stops
playing early and discards. <b>Use as a weak baseline only</b> — it never improves.</p>
<p>🟢 Instant decisions &nbsp;|&nbsp; 🔴 Very weak</p>

<h3 style="color:#ffe066;">Heuristic Agent</h3>
<p>Uses hand-crafted priority rules: stockpile plays first, then combo setups,
wild card conservation, and smart discarding. Performs consistently well with
no training required.</p>
<p>🟢 Instant decisions &nbsp;|&nbsp; 🟡 Strong baseline &nbsp;|&nbsp; 🟢 Best for training partners</p>

<h3 style="color:#ffe066;">MCTS-N (Monte Carlo Search)</h3>
<p>For each possible action, runs <b>N random game simulations</b> to the end and
picks the action with the best win rate. Requires <b>no training</b> — it thinks
at decision time.</p>
<ul>
<li><b>MCTS-5</b>  — fast, use for training benchmarks</li>
<li><b>MCTS-15</b> — balanced speed/strength, recommended for gameplay</li>
<li><b>MCTS-30</b> — strong, slight delay per move</li>
<li><b>MCTS-50</b> — very strong, ~1-2 s per decision</li>
</ul>
<p>⚠️ <b>Training with MCTS is very slow</b> — each episode takes much longer than
with Heuristic or Random. Use MCTS-5 or MCTS-10 if you must train with it,
or just benchmark it in AI vs AI mode instead.</p>
<p>🟡 Moderate decisions &nbsp;|&nbsp; 🟢 Strongest no-training agent</p>

<h3 style="color:#ffe066;">DQN (Deep Q-Network)</h3>
<p>A neural network trained via reinforcement learning. Learns from game experience
over many episodes. <b>Needs extensive training to become competent</b> —
at least 50,000+ episodes for meaningful improvement.</p>
<p>Tips for better DQN training:</p>
<ul>
<li>Train for 10,000+ episodes minimum</li>
<li>Use Heuristic as the opponent (tougher teacher than Random)</li>
<li>DQN slowly improves — check the win rate trend chart in the training screen</li>
<li>Without PyTorch installed, DQN falls back to random play</li>
</ul>
<p>🔴 Requires training &nbsp;|&nbsp; 🟡 Can become strong with enough episodes</p>

<h3 style="color:#ffe066;">Recommended Match-ups</h3>
<ul>
<li><b>Best for learning the game:</b> Human vs Heuristic</li>
<li><b>Best AI challenge:</b> Human vs MCTS-30</li>
<li><b>Best for watching AI:</b> Heuristic vs MCTS-15</li>
<li><b>Best for training DQN:</b> DQN vs Heuristic (10k+ episodes)</li>
<li><b>Fastest training stats:</b> Heuristic vs Random</li>
</ul>

<h3 style="color:#ffe066;">Statistics Screen</h3>
<p>Open <b>View Statistics</b> from the main menu to see graphs for every
matchup you've played or trained. Data is saved automatically — no file
loading needed. Three chart tabs are available:</p>
<ul>
<li><b>Win Rate Trend</b> — rolling win rate over games for the selected matchup</li>
<li><b>Game Length</b> — histogram of how many turns games last</li>
<li><b>All Matchups</b> — side-by-side win % comparison across all recorded matchups</li>
</ul>
""",
}


class RulesScreen(QWidget):
    return_to_menu = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COLOR_UI_BG};")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Title + back button
        title_row = QHBoxLayout()
        title = QLabel("📖  Rules & Help Guide")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{COLOR_UI_ACCENT};")
        title_row.addWidget(title)
        title_row.addStretch()
        back = QPushButton("← Back to Menu")
        back.setStyleSheet(BTN)
        back.clicked.connect(self.return_to_menu.emit)
        title_row.addWidget(back)
        root.addLayout(title_row)

        # Tab buttons
        tab_row = QHBoxLayout()
        tab_row.setSpacing(8)
        self._tab_btns = {}
        for section in SECTIONS:
            btn = QPushButton(section)
            btn.setStyleSheet(BTN)
            btn.setCheckable(True)
            btn.clicked.connect(self._make_tab_handler(section))
            tab_row.addWidget(btn)
            self._tab_btns[section] = btn
        tab_row.addStretch()
        root.addLayout(tab_row)

        # Divider
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background:{COLOR_UI_ACCENT}; max-height:1px;")
        root.addWidget(line)

        # Content browser
        self._browser = QTextBrowser()
        self._browser.setStyleSheet(f"""
            QTextBrowser {{
                background:{COLOR_UI_PANEL}; color:{COLOR_UI_TEXT};
                border:1px solid #333344; border-radius:8px;
                padding:16px; font-size:12px; line-height:1.6;
            }}
            QScrollBar:vertical {{ background:#1e1e2e; width:10px; }}
            QScrollBar::handle:vertical {{ background:#4a4a7a; border-radius:5px; }}
        """)
        self._browser.setOpenExternalLinks(False)
        root.addWidget(self._browser, stretch=1)

        # Show first section by default
        first = list(SECTIONS.keys())[0]
        self._show_section(first)

    def _make_tab_handler(self, section):
        def handler():
            self._show_section(section)
        return handler

    def _show_section(self, section: str):
        for name, btn in self._tab_btns.items():
            btn.setChecked(name == section)
        self._browser.setHtml(f"""
        <html><body style="
            background:{COLOR_UI_PANEL}; color:{COLOR_UI_TEXT};
            font-family:Arial,sans-serif; font-size:13px; line-height:1.7;
            margin:0; padding:8px;
        ">
        {SECTIONS.get(section, '')}
        </body></html>
        """)
        self._browser.verticalScrollBar().setValue(0)