"""
BoardView – the main game playing area widget.

Interaction model
-----------------
PLAY phase:
  1. Click a hand card / stockpile / discard-pile-top  → selects it (blue highlight)
  2. Click a building pile                              → plays the card there
  3. Click the same card again                         → deselects

DISCARD / end-turn:
  • With a hand card selected, click any of YOUR discard piles → discard & end turn
  • OR click the "End Turn" button in the side panel           → auto-discard
"""

from typing import Optional, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from game.game_state import GameState, Action
from game.card import Card
from game.player import Player
from gui.card_widget import CardWidget
from utils.config import (
    COLOR_TABLE, COLOR_TABLE_ALT, COLOR_UI_BG, COLOR_UI_PANEL,
    COLOR_UI_TEXT, COLOR_UI_ACCENT, COLOR_BUILD_BG,
    CARD_W, CARD_H, CARD_SPACING,
    COLOR_LEGAL_GLOW, COLOR_HIGHLIGHT,
)


def _label(text: str, bold: bool = False, color: str = COLOR_UI_TEXT,
           size: int = 10) -> QLabel:
    lbl = QLabel(text)
    font = QFont("Arial", size)
    font.setBold(bold)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; background: transparent;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return lbl


def _hline() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background: {COLOR_UI_ACCENT}; max-height: 1px;")
    return f


class ZoneWidget(QFrame):
    def __init__(self, title: str, bg: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {bg}; border-radius: 8px;")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 4)
        self._layout.setSpacing(4)
        if title:
            lbl = _label(title, bold=True, size=9, color=COLOR_UI_ACCENT)
            self._layout.addWidget(lbl)

    def inner_layout(self):
        return self._layout


class BoardView(QWidget):
    """
    Interactive board.  Emits ``action_requested`` with a fully-formed Action
    when the human player completes a valid interaction.
    """

    action_requested = pyqtSignal(object)   # Action

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: Optional[GameState] = None
        self.human_player_idx: int = 0
        self.interactive: bool = True

        # Selection state
        self._selected_card: Optional[Card] = None
        self._selected_source: Optional[str] = None  # 'hand', 'stockpile', 'discard_N'

        # Track all live CardWidgets so we can update them safely.
        # IMPORTANT: always call _card_widgets.clear() BEFORE _clear_layout()
        # so we never hold stale C++ references.
        self._card_widgets: dict = {}

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setStyleSheet(f"background: {COLOR_TABLE};")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # Opponent zone
        self._opp_zone = ZoneWidget("OPPONENT", COLOR_TABLE_ALT)
        self._opp_row = QHBoxLayout()
        self._opp_row.setSpacing(CARD_SPACING)
        self._opp_zone.inner_layout().addLayout(self._opp_row)
        root.addWidget(self._opp_zone)

        root.addWidget(_hline())

        # Building piles
        self._build_zone = ZoneWidget("BUILDING PILES  (play cards here)", COLOR_BUILD_BG)
        self._build_row = QHBoxLayout()
        self._build_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._build_row.setSpacing(20)
        self._build_zone.inner_layout().addLayout(self._build_row)
        root.addWidget(self._build_zone)

        root.addWidget(_hline())

        # Player zone
        self._player_zone = ZoneWidget("YOUR AREA", COLOR_TABLE_ALT)
        self._player_top_row = QHBoxLayout()
        self._player_top_row.setSpacing(CARD_SPACING)
        self._player_hand_row = QHBoxLayout()
        self._player_hand_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._player_hand_row.setSpacing(CARD_SPACING)

        # End turn button inside the player zone for easy access
        self._end_turn_btn = QPushButton("⏭  End Turn  (discard selected card)")
        self._end_turn_btn.setStyleSheet(f"""
            QPushButton {{
                background: #336633; color: {COLOR_UI_TEXT};
                border: none; border-radius: 6px;
                padding: 6px 18px; font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #449944; }}
            QPushButton:disabled {{ background: #222233; color: #555566; }}
        """)
        self._end_turn_btn.clicked.connect(self._on_end_turn_clicked)

        self._player_zone.inner_layout().addLayout(self._player_top_row)
        self._player_zone.inner_layout().addLayout(self._player_hand_row)
        self._player_zone.inner_layout().addWidget(
            self._end_turn_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._player_zone)

        # Status bar
        self._status_label = _label("", size=11, color="#ffe066")
        root.addWidget(self._status_label)

    # ── Public interface ───────────────────────────────────────────────────────
    def set_state(self, state: GameState, human_idx: int = 0):
        self.state = state
        self.human_player_idx = human_idx
        self._selected_card = None
        self._selected_source = None
        self._refresh()

    def refresh(self):
        if self.state:
            self._refresh()

    def set_interactive(self, value: bool):
        self.interactive = value
        self._end_turn_btn.setEnabled(value)
        self._safe_update_cursors()

    def set_status(self, msg: str):
        self._status_label.setText(msg)

    def clear_selection(self):
        self._selected_card = None
        self._selected_source = None
        self._refresh()

    # ── Full board refresh ─────────────────────────────────────────────────────
    def _refresh(self):
        if not self.state:
            return

        # !! Clear the reference dict BEFORE deleting any widgets.
        # This prevents _update_interactivity (called later) from touching
        # already-deleted C++ CardWidget objects → RuntimeError.
        self._card_widgets.clear()

        self._clear_layout(self._opp_row)
        self._clear_layout(self._build_row)
        self._clear_layout(self._player_top_row)
        self._clear_layout(self._player_hand_row)

        player = self.state.players[self.human_player_idx]
        opp    = self.state.players[1 - self.human_player_idx]

        self._render_opponent_zone(opp)
        self._render_building_piles()
        self._render_player_zone(player)
        self._safe_update_cursors()
        self._update_end_turn_btn(player)

    # ── Zone renderers ─────────────────────────────────────────────────────────
    def _render_opponent_zone(self, opp: Player):
        sp = self._make_card_widget(opp.stockpile_top, face_up=True, clickable=False)
        sp.empty_label = "DONE!" if opp.stockpile_size == 0 else ""
        self._opp_row.addLayout(
            self._labelled(f"Stock ({opp.stockpile_size})", sp))

        self._opp_row.addSpacing(12)

        for i, top in enumerate(opp.discard_tops()):
            w = self._make_card_widget(top, face_up=True, clickable=False)
            w.empty_label = f"D{i+1}"
            self._opp_row.addLayout(
                self._labelled(f"Disc {i+1} ({opp.discard_pile_size(i)})", w))

        self._opp_row.addStretch()
        self._opp_row.addWidget(_label(f"✋ {opp.hand_size}", bold=True, size=12))

    def _render_building_piles(self):
        for i in range(self.state.MAX_BUILDING_PILES):
            pile = self.state.building_piles[i]
            if pile:
                top_card = pile[-1]
                next_v   = self.state.pile_needed_value(i)
                w = self._make_card_widget(top_card, face_up=True, clickable=True)
                info = f"[{len(pile)}]  next→{next_v}" if next_v <= 12 else "Complete!"
            else:
                w = CardWidget(card=None)
                w.empty_label = "ACE?"
                info = "empty → ACE"

            # Highlight pile if selected card can play here
            if self._selected_card is not None:
                can_play = self.state.can_play_card_to_pile(self._selected_card, i)
                w.set_legal_target(can_play)

            w.clicked.connect(self._make_build_handler(i))
            self._build_row.addLayout(self._labelled(info, w))
            self._card_widgets[f'build_{i}'] = w

    def _render_player_zone(self, player: Player):
        # Stockpile
        sp = self._make_card_widget(player.stockpile_top, face_up=True, clickable=True)
        sp.empty_label = "🏆" if player.stockpile_size == 0 else ""
        if self._selected_card == player.stockpile_top and self._selected_source == 'stockpile':
            sp.set_selected(True)
        sp.clicked.connect(self._on_stockpile_clicked)
        self._player_top_row.addLayout(
            self._labelled(f"Stock ({player.stockpile_size})", sp))
        self._card_widgets['player_stock'] = sp

        self._player_top_row.addSpacing(12)

        # Player discard piles
        for i, top in enumerate(player.discard_tops()):
            w = self._make_card_widget(top, face_up=True, clickable=True)
            w.empty_label = f"D{i+1}"

            # Highlight discard piles when a hand card is selected (for discarding)
            if self._selected_source == 'hand' and self._selected_card is not None:
                w.set_legal_target(True)   # any discard pile is valid

            w.clicked.connect(self._make_discard_handler(i))
            self._player_top_row.addLayout(
                self._labelled(f"Disc {i+1} ({player.discard_pile_size(i)})", w))
            self._card_widgets[f'player_discard_{i}'] = w

        self._player_top_row.addStretch()

        # Hand cards
        for idx, card in enumerate(player.hand):
            hw = self._make_card_widget(card, face_up=True, clickable=True)
            if self._selected_card == card and self._selected_source == 'hand':
                hw.set_selected(True)
            hw.clicked.connect(self._make_hand_handler(idx))
            self._player_hand_row.addWidget(hw)
            self._card_widgets[f'hand_{idx}'] = hw

    # ── Widget factory ─────────────────────────────────────────────────────────
    def _make_card_widget(self, card: Optional[Card], face_up: bool,
                           clickable: bool) -> CardWidget:
        w = CardWidget(card, face_up)
        if not clickable:
            w.setCursor(Qt.CursorShape.ArrowCursor)
        return w

    def _labelled(self, text: str, widget: QWidget) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(_label(text, size=8), alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignCenter)
        return layout

    # ── Click handler factories (closures avoid late-binding bugs) ─────────────
    def _make_hand_handler(self, idx: int):
        def handler(card):
            if self.interactive and self.state:
                self._on_hand_clicked(idx)
        return handler

    def _make_build_handler(self, pile_idx: int):
        def handler(card):
            if self.interactive and self.state:
                self._on_build_clicked(pile_idx)
        return handler

    def _make_discard_handler(self, pile_idx: int):
        def handler(card):
            if self.interactive and self.state:
                self._on_discard_pile_clicked(pile_idx)
        return handler

    # ── Interaction logic ──────────────────────────────────────────────────────
    def _on_hand_clicked(self, idx: int):
        player = self.state.current_player
        if idx >= len(player.hand):
            return
        card = player.hand[idx]

        # Toggle selection
        if self._selected_card == card and self._selected_source == 'hand':
            self._selected_card = None
            self._selected_source = None
            self.set_status("Card deselected.")
        else:
            self._selected_card = card
            self._selected_source = 'hand'
            has_play = any(
                self.state.can_play_card_to_pile(card, i)
                for i in range(self.state.MAX_BUILDING_PILES)
            )
            if has_play:
                self.set_status(
                    f"Selected {card.display()} — click a highlighted building pile to play, "
                    "or click a DISCARD pile to discard & end turn.")
            else:
                self.set_status(
                    f"Selected {card.display()} — click a DISCARD PILE to discard & end turn.")
        self._refresh()

    def _on_stockpile_clicked(self, card):
        if not self.state or not self.interactive:
            return
        player = self.state.current_player
        top = player.stockpile_top
        if top is None:
            return
        if self._selected_card == top and self._selected_source == 'stockpile':
            self._selected_card = None
            self._selected_source = None
        else:
            self._selected_card = top
            self._selected_source = 'stockpile'
            self.set_status(f"Stockpile top {top.display()} selected — click a building pile.")
        self._refresh()

    def _on_discard_pile_clicked(self, pile_idx: int):
        """
        Two uses:
          A) A hand card is selected → discard it here & end turn.
          B) A discard top is selected (for playing to build) — handled below.
          C) Nothing selected → select this discard top for playing.
        """
        if not self.state or not self.interactive:
            return
        player = self.state.current_player

        # A) Discard selected hand card → end turn
        if self._selected_source == 'hand' and self._selected_card is not None:
            action = Action('discard', self._selected_card, None, pile_idx)
            self._selected_card = None
            self._selected_source = None
            self.action_requested.emit(action)
            return

        # B) Nothing selected yet → select this discard pile top for playing
        top = player.discard_piles[pile_idx][-1] if player.discard_piles[pile_idx] else None
        if top is None:
            self.set_status(f"Discard pile {pile_idx+1} is empty.")
            return

        if self._selected_card == top and self._selected_source == f'discard_{pile_idx}':
            self._selected_card = None
            self._selected_source = None
        else:
            self._selected_card = top
            self._selected_source = f'discard_{pile_idx}'
            self.set_status(f"Discard top {top.display()} selected — click a building pile.")
        self._refresh()

    def _on_build_clicked(self, pile_idx: int):
        if not self.state or not self.interactive:
            return
        if self._selected_card is None:
            self.set_status("Select a card first (from your hand, stockpile, or discard pile).")
            return

        card   = self._selected_card
        source = self._selected_source
        wv     = self.state._resolve_wild_value(card, pile_idx) if card.is_wild else None

        if source == 'hand':
            action = Action('play_hand', card, None, pile_idx, wv)
        elif source == 'stockpile':
            action = Action('play_stockpile', card, None, pile_idx, wv)
        elif source and source.startswith('discard_'):
            src_pile = int(source.split('_')[1])
            action = Action('play_discard', card, src_pile, pile_idx, wv)
        else:
            return

        self._selected_card  = None
        self._selected_source = None
        self.action_requested.emit(action)

    def _on_end_turn_clicked(self):
        """
        End turn button inside the board.
        If a hand card is already selected, discard it to pile 0.
        Otherwise auto-pick the least useful hand card.
        """
        if not self.state or not self.interactive:
            return
        player = self.state.current_player
        if not player.hand:
            self.set_status("No hand cards to discard!")
            return

        card = self._selected_card if (self._selected_source == 'hand' and
                                        self._selected_card in player.hand) \
               else player.hand[0]

        # Find the best discard pile (prefer non-empty so we can stack)
        pile_idx = 0
        for i, pile in enumerate(player.discard_piles):
            if not pile:
                pile_idx = i
                break

        action = Action('discard', card, None, pile_idx)
        self._selected_card  = None
        self._selected_source = None
        self.action_requested.emit(action)

    # ── UI update helpers ──────────────────────────────────────────────────────
    def _update_end_turn_btn(self, player: Player):
        is_human = self.state and self.state.current_player_idx == self.human_player_idx
        self._end_turn_btn.setEnabled(is_human and self.interactive and bool(player.hand))

    def _safe_update_cursors(self):
        """Update cursors on all live card widgets, ignoring any stale C++ objects."""
        cursor = (Qt.CursorShape.PointingHandCursor if self.interactive
                  else Qt.CursorShape.ArrowCursor)
        stale = []
        for key, w in self._card_widgets.items():
            try:
                w.setCursor(cursor)
            except RuntimeError:
                stale.append(key)
        for key in stale:
            del self._card_widgets[key]

    # ── Layout utilities ───────────────────────────────────────────────────────
    @staticmethod
    def _clear_layout(layout):
        """Remove and schedule deletion of all items in *layout* recursively."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)   # immediate reparenting; deleteLater follows
                w.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    BoardView._clear_layout(sub)