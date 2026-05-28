"""
Core game state and action logic for Spite Analysis.

Turn flow
---------
1. start of turn  → cards already drawn by _switch_turn from previous turn
2. play phase     → player calls apply_action() with play_* actions any number of times
3. end turn       → player calls apply_action() with a 'discard' action → turn switches
                    OR hand becomes empty → turn auto-switches

Auto-end edge cases
-------------------
If a player's hand is empty AND there are no play actions available (e.g. all hand
cards were played and stockpile/discard plays are also exhausted), the turn is
switched automatically so the game never stalls.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from game.card import Card, Suit
from game.deck import Deck
from game.player import Player


# ─────────────────────────────────────────────────────────────────────────────
# Action dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Action:
    """A single discrete game action."""

    # 'play_hand'      – play a card from the current player's hand
    # 'play_stockpile' – play the stockpile top
    # 'play_discard'   – play the top of one of the player's discard piles
    # 'discard'        – discard a hand card onto a personal discard pile → ends turn
    action_type: str

    card: Optional[Card] = field(default=None, compare=False)
    source_pile: Optional[int] = None    # discard pile idx (for play_discard)
    target_pile: int = 0                 # building pile (0-3) or discard pile idx
    wild_value: Optional[int] = None     # value the wild card will represent

    def __repr__(self):
        if self.action_type == 'discard':
            return f"DISCARD {self.card} → discard[{self.target_pile}]"
        src = (f"discard[{self.source_pile}]"
               if self.source_pile is not None
               else self.action_type.split('_')[1])
        return (f"PLAY {self.card} from {src} → build[{self.target_pile}]"
                + (f" as {self.wild_value}" if self.wild_value else ""))


# ─────────────────────────────────────────────────────────────────────────────
# GameState
# ─────────────────────────────────────────────────────────────────────────────

class GameState:
    """Complete mutable game state for Spite Analysis (2 players)."""

    PHASE_PLAY    = 'play'
    PHASE_DISCARD = 'discard'

    MAX_BUILDING_PILES = 4
    STOCKPILE_DEAL     = 13

    def __init__(self, seed: Optional[int] = None,
                 player_names: Tuple[str, str] = ("Human", "AI")):
        self.seed = seed
        self.player_names = player_names

        self.deck = Deck(seed)
        self.players: List[Player] = [
            Player(0, player_names[0]),
            Player(1, player_names[1]),
        ]
        # Four fixed building pile slots; empty list = needs Ace
        self.building_piles: List[List[Card]] = [[], [], [], []]

        self.current_player_idx: int = 0
        self.phase: str = self.PHASE_PLAY
        self.turn_number: int = 1
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.completed_sequences: int = 0

        self.action_history: List[Action] = []
        self.stats: Dict[str, Any] = {'total_plays': 0, 'total_discards': 0}

        self._deal()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _deal(self):
        for player in self.players:
            player.stockpile = self.deck.draw_many(self.STOCKPILE_DEAL)
        for player in self.players:
            for card in self.deck.draw_many(Player.MAX_HAND_SIZE):
                player.add_to_hand(card)
        self.phase = self.PHASE_PLAY

    # ── Turn management ────────────────────────────────────────────────────────
    def _draw_for_current_player(self):
        player = self.current_player
        needed = player.cards_to_draw()
        for card in self.deck.draw_many(needed):
            player.add_to_hand(card)

    def _switch_turn(self):
        """Advance to next player, draw cards, reset phase."""
        self.current_player_idx = 1 - self.current_player_idx
        self.turn_number += 1
        self.phase = self.PHASE_PLAY
        self._draw_for_current_player()

    # ── Properties ─────────────────────────────────────────────────────────────
    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def opponent(self) -> Player:
        return self.players[1 - self.current_player_idx]

    # ── Building pile helpers ──────────────────────────────────────────────────
    def pile_top_value(self, pile_idx: int) -> int:
        """Effective top value (0 = empty pile, i.e. needs Ace)."""
        pile = self.building_piles[pile_idx]
        return pile[-1].effective_value() if pile else 0

    def pile_needed_value(self, pile_idx: int) -> int:
        """Next required value for building pile (1 = Ace for empty pile)."""
        return self.pile_top_value(pile_idx) + 1

    def can_play_card_to_pile(self, card: Card, pile_idx: int,
                               wild_value: Optional[int] = None) -> bool:
        if pile_idx < 0 or pile_idx >= self.MAX_BUILDING_PILES:
            return False
        needed = self.pile_needed_value(pile_idx)
        if needed > 12:
            return False
        if card.is_wild:
            wv = wild_value if wild_value is not None else needed
            return card.can_represent(wv) and wv == needed
        return card.base_value == needed

    def _resolve_wild_value(self, card: Card, pile_idx: int,
                             hint: Optional[int] = None) -> Optional[int]:
        """Return the wild value for card on pile_idx, or None if impossible."""
        if not card.is_wild:
            return None
        needed = self.pile_needed_value(pile_idx)
        target = hint if hint is not None else needed
        if card.can_represent(target) and target == needed:
            return target
        return None

    def _apply_card_to_pile(self, card: Card, pile_idx: int,
                             wild_value: Optional[int] = None) -> bool:
        """Place card on building pile. Returns True on success."""
        needed = self.pile_needed_value(pile_idx)
        if needed > 12:
            return False

        if card.is_wild:
            wv = wild_value if wild_value is not None else needed
            if not card.can_represent(wv) or wv != needed:
                return False
            card = card.copy()
            card.wild_as = wv
        else:
            if card.base_value != needed:
                return False

        self.building_piles[pile_idx].append(card)
        self.stats['total_plays'] += 1

        # Complete at Queen (12)
        if card.effective_value() == 12:
            self.deck.return_completed_pile(self.building_piles[pile_idx])
            self.building_piles[pile_idx] = []
            self.completed_sequences += 1

        return True

    # ── Action generation ──────────────────────────────────────────────────────
    def get_play_actions(self) -> List[Action]:
        """All valid PLAY actions for the current player."""
        if self.game_over:
            return []

        actions: List[Action] = []
        player = self.current_player

        def _add(card: Card, atype: str, src_pile: Optional[int] = None):
            for pi in range(self.MAX_BUILDING_PILES):
                if card.is_wild:
                    wv = self._resolve_wild_value(card, pi)
                    if wv is not None:
                        actions.append(Action(atype, card, src_pile, pi, wv))
                else:
                    if self.can_play_card_to_pile(card, pi):
                        actions.append(Action(atype, card, src_pile, pi))

        for card in player.hand:
            _add(card, 'play_hand')

        if player.stockpile_top:
            _add(player.stockpile_top, 'play_stockpile')

        for pi, top in enumerate(player.discard_tops()):
            if top is not None:
                _add(top, 'play_discard', pi)

        return actions

    def get_discard_actions(self) -> List[Action]:
        """All valid DISCARD actions (hand card → personal discard pile)."""
        if self.game_over:
            return []
        player = self.current_player
        return [
            Action('discard', card, None, di)
            for card in player.hand
            for di in range(Player.MAX_DISCARD_PILES)
        ]

    def get_valid_actions(self) -> List[Action]:
        """All valid actions for the current player."""
        return self.get_play_actions() + self.get_discard_actions()

    # ── Action application ─────────────────────────────────────────────────────
    def apply_action(self, action: Action) -> Tuple[bool, Optional[int]]:
        """
        Apply action to game state.
        Returns (success, winner_id).
        """
        if self.game_over:
            return False, self.winner

        player = self.current_player

        # ── Play from hand ─────────────────────────────────────────────────────
        if action.action_type == 'play_hand':
            if action.card not in player.hand:
                return False, None
            player.remove_from_hand(action.card)
            if not self._apply_card_to_pile(action.card, action.target_pile, action.wild_value):
                player.add_to_hand(action.card)
                return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over:
                self._auto_end_if_stuck(player)
            return True, self.winner

        # ── Play from stockpile ────────────────────────────────────────────────
        elif action.action_type == 'play_stockpile':
            if not player.stockpile_top:
                return False, None
            card = player.pop_stockpile()
            if not self._apply_card_to_pile(card, action.target_pile, action.wild_value):
                player.stockpile.append(card)
                return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over:
                self._auto_end_if_stuck(player)
            return True, self.winner

        # ── Play from discard pile ─────────────────────────────────────────────
        elif action.action_type == 'play_discard':
            if action.source_pile is None:
                return False, None
            card = player.pop_discard(action.source_pile)
            if card is None:
                return False, None
            if not self._apply_card_to_pile(card, action.target_pile, action.wild_value):
                player.push_discard(card, action.source_pile)
                return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over:
                self._auto_end_if_stuck(player)
            return True, self.winner

        # ── Discard → ends turn ────────────────────────────────────────────────
        elif action.action_type == 'discard':
            if action.card not in player.hand:
                return False, None
            player.remove_from_hand(action.card)
            player.push_discard(action.card, action.target_pile)
            self.action_history.append(action)
            self.stats['total_discards'] += 1
            self._switch_turn()
            return True, self.winner

        return False, None

    # ── Win / auto-end helpers ─────────────────────────────────────────────────
    def _check_win(self):
        if self.current_player.has_won:
            self.game_over = True
            self.winner = self.current_player_idx

    def _auto_end_if_stuck(self, player: Player):
        """
        If the player has no hand cards AND no play actions available,
        switch the turn automatically so the game never stalls.
        This is an edge case (played all 5 hand cards and stockpile/discard
        have nothing playable), but it must be handled gracefully.
        """
        if not player.hand and not self.get_play_actions():
            self._switch_turn()

    # ── RL environment interface ───────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> 'GameState':
        new_seed = seed if seed is not None else self.seed
        self.__init__(new_seed, self.player_names)
        return self

    def is_terminal(self) -> bool:
        return self.game_over

    def get_observation(self) -> Dict[str, Any]:
        player = self.current_player
        opp    = self.opponent
        return {
            'hand':              [c.base_value for c in player.hand],
            'hand_wild':         [1 if c.is_wild else 0 for c in player.hand],
            'stockpile_top':     player.stockpile_top.base_value if player.stockpile_top else 0,
            'stockpile_top_wild': 1 if (player.stockpile_top and player.stockpile_top.is_wild) else 0,
            'stockpile_size':    player.stockpile_size,
            'discard_tops':      [(c.base_value if c else 0) for c in player.discard_tops()],
            'build_tops':        [self.pile_top_value(i) for i in range(self.MAX_BUILDING_PILES)],
            'opp_stockpile_top': opp.stockpile_top.base_value if opp.stockpile_top else 0,
            'opp_stockpile_size': opp.stockpile_size,
            'opp_discard_tops':  [(c.base_value if c else 0) for c in opp.discard_tops()],
            'opp_hand_size':     opp.hand_size,
            'deck_remaining':    self.deck.remaining,
            'turn_number':       self.turn_number,
            'current_player':    self.current_player_idx,
            'phase':             0 if self.phase == self.PHASE_PLAY else 1,
        }

    def get_reward(self, player_id: int) -> float:
        if self.game_over:
            return 1.0 if self.winner == player_id else -1.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        player = self.current_player
        opp    = self.opponent
        return {
            'turn':     self.turn_number,
            'current_player': self.current_player_idx,
            'phase':    self.phase,
            'build_piles': [[c.display() for c in pile] for pile in self.building_piles],
            'build_tops':  [self.pile_top_value(i) for i in range(self.MAX_BUILDING_PILES)],
            'player': {
                'hand':          [c.display() for c in player.hand],
                'stockpile_top': player.stockpile_top.display() if player.stockpile_top else None,
                'stockpile_size': player.stockpile_size,
                'discard_tops':  [c.display() if c else None for c in player.discard_tops()],
            },
            'opponent': {
                'hand_size':     opp.hand_size,
                'stockpile_top': opp.stockpile_top.display() if opp.stockpile_top else None,
                'stockpile_size': opp.stockpile_size,
                'discard_tops':  [c.display() if c else None for c in opp.discard_tops()],
            },
            'deck_remaining':      self.deck.remaining,
            'completed_sequences': self.completed_sequences,
            'game_over': self.game_over,
            'winner':    self.winner,
        }

    def copy(self) -> 'GameState':
        return copy.deepcopy(self)

    def __repr__(self):
        return (f"GameState(turn={self.turn_number}, "
                f"player={self.current_player.name}, phase={self.phase})")