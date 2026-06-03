"""
Core game state for Spite Analysis.

Key rules implemented here
---------------------------
1. Hand refill – play all 5 cards → draw 5 fresh ones, turn continues.
   If deck is exhausted → turn passes automatically.

2. Building-pile access (unlock rule)
   A player may only play NON-ACE cards to building piles once they are
   "unlocked".  A player becomes unlocked when:
     a. They personally play an Ace to start a building pile, OR
     b. The opponent has cumulatively started 4 or more piles (placed 4 Aces).

3. No draws – a stall counter detects when neither player can move; the
   player with the smaller stockpile wins (player 0 wins ties).
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from game.card import Card, Suit
from game.deck import Deck
from game.player import Player


@dataclass
class Action:
    action_type: str   # play_hand | play_stockpile | play_discard | discard
    card: Optional[Card] = field(default=None, compare=False)
    source_pile: Optional[int] = None
    target_pile: int = 0
    wild_value: Optional[int] = None

    def __repr__(self):
        if self.action_type == 'discard':
            return f"DISCARD {self.card} → discard[{self.target_pile}]"
        src = (f"discard[{self.source_pile}]" if self.source_pile is not None
               else self.action_type.split('_')[1])
        return (f"PLAY {self.card} from {src} → build[{self.target_pile}]"
                + (f" as {self.wild_value}" if self.wild_value else ""))


class GameState:
    PHASE_PLAY    = 'play'
    PHASE_DISCARD = 'discard'
    MAX_BUILDING_PILES = 4
    STOCKPILE_DEAL     = 13

    def __init__(self, seed: Optional[int] = None,
                 player_names: Tuple[str, str] = ("Human", "AI")):
        self.seed = seed
        self.player_names = player_names
        self.deck = Deck(seed)
        self.players: List[Player] = [Player(0, player_names[0]),
                                       Player(1, player_names[1])]
        self.building_piles: List[List[Card]] = [[], [], [], []]
        self.current_player_idx: int = 0
        self.phase: str = self.PHASE_PLAY
        self.turn_number: int = 1
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.completed_sequences: int = 0
        self.hand_refills: int = 0

        # ── Unlock rule state ──────────────────────────────────────────────────
        # pile_starts_by[i] = how many Aces player i has ever played to building piles
        self.pile_starts_by: List[int] = [0, 0]
        # player_unlocked[i] = may player i play non-Ace cards to building piles?
        self.player_unlocked: List[bool] = [False, False]

        # ── Stall / no-draw detection ──────────────────────────────────────────
        # Incremented every time a turn passes with zero productive actions.
        # When it reaches 4 (both players stuck twice each) the game is
        # resolved by stockpile size so there is never a draw.
        self._stall_ticks: int = 0

        self.action_history: List[Action] = []
        self.stats: Dict[str, Any] = {'total_plays': 0, 'total_discards': 0}
        self._deal()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _deal(self):
        for p in self.players:
            p.stockpile = self.deck.draw_many(self.STOCKPILE_DEAL)
        for p in self.players:
            for c in self.deck.draw_many(Player.MAX_HAND_SIZE):
                p.add_to_hand(c)

    # ── Turn management ────────────────────────────────────────────────────────
    def _draw_for_current_player(self):
        p = self.current_player
        for c in self.deck.draw_many(p.cards_to_draw()):
            p.add_to_hand(c)

    def _switch_turn(self):
        self.current_player_idx = 1 - self.current_player_idx
        self.turn_number += 1
        self.phase = self.PHASE_PLAY
        self._draw_for_current_player()
        # If the new player has NO hand cards AND no play actions (deck exhausted),
        # detect the stall immediately so run_game never asks an agent to act
        # when it genuinely cannot — preventing the "returned None" warnings.
        if not self.current_player.hand and not self.get_play_actions():
            self._stall_ticks += 1
            self._maybe_force_winner()

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]

    @property
    def opponent(self) -> Player:
        return self.players[1 - self.current_player_idx]

    # ── Building pile helpers ──────────────────────────────────────────────────
    def pile_top_value(self, i: int) -> int:
        p = self.building_piles[i]
        return p[-1].effective_value() if p else 0

    def pile_needed_value(self, i: int) -> int:
        return self.pile_top_value(i) + 1

    def can_play_card_to_pile(self, card: Card, pi: int,
                               wild_value: Optional[int] = None) -> bool:
        """Return True if card can legally be played to building pile pi."""
        if pi < 0 or pi >= self.MAX_BUILDING_PILES:
            return False
        needed = self.pile_needed_value(pi)
        if needed > 12:
            return False

        # Determine the effective play value
        if card.is_wild:
            wv = wild_value if wild_value is not None else needed
            if not (card.can_represent(wv) and wv == needed):
                return False
            play_value = wv
        else:
            if card.base_value != needed:
                return False
            play_value = card.base_value

        # ── Unlock rule ────────────────────────────────────────────────────────
        # Playing an Ace (value 1) is ALWAYS permitted – it is how you unlock.
        # Playing any other value requires the player to be unlocked first.
        if play_value != 1 and not self.player_unlocked[self.current_player_idx]:
            return False

        return True

    def _resolve_wild_value(self, card: Card, pi: int,
                             hint: Optional[int] = None) -> Optional[int]:
        if not card.is_wild:
            return None
        needed = self.pile_needed_value(pi)
        target = hint if hint is not None else needed
        return target if (card.can_represent(target) and target == needed) else None

    def _apply_to_pile(self, card: Card, pi: int,
                        wild_value: Optional[int] = None) -> bool:
        needed = self.pile_needed_value(pi)
        if needed > 12:
            return False

        if card.is_wild:
            wv = wild_value if wild_value is not None else needed
            if not card.can_represent(wv) or wv != needed:
                return False
            card = card.copy(); card.wild_as = wv
        else:
            if card.base_value != needed:
                return False

        self.building_piles[pi].append(card)
        self.stats['total_plays'] += 1

        # ── Track Ace / unlock logic ───────────────────────────────────────────
        if card.effective_value() == 1:
            pid = self.current_player_idx
            self.pile_starts_by[pid] += 1
            # Playing an Ace unlocks yourself
            self.player_unlocked[pid] = True
            # If opponent has placed 4+ Aces, opponent becomes unlocked too
            # (but here we check: have WE placed 4, unlocking the opponent?)
            opp = 1 - pid
            if self.pile_starts_by[pid] >= 4:
                self.player_unlocked[opp] = True

        # ── Pile completion at Queen (12) ──────────────────────────────────────
        if card.effective_value() == 12:
            self.deck.return_completed_pile(self.building_piles[pi])
            self.building_piles[pi] = []
            self.completed_sequences += 1

        # Any successful play resets the stall counter
        self._stall_ticks = 0
        return True

    # ── Action generation ──────────────────────────────────────────────────────
    def get_play_actions(self) -> List[Action]:
        if self.game_over:
            return []
        acts: List[Action] = []
        p = self.current_player

        def add(card, atype, src=None):
            for pi in range(self.MAX_BUILDING_PILES):
                if card.is_wild:
                    wv = self._resolve_wild_value(card, pi)
                    if wv is not None and self.can_play_card_to_pile(card, pi, wv):
                        acts.append(Action(atype, card, src, pi, wv))
                elif self.can_play_card_to_pile(card, pi):
                    acts.append(Action(atype, card, src, pi))

        for c in p.hand:
            add(c, 'play_hand')
        if p.stockpile_top:
            add(p.stockpile_top, 'play_stockpile')
        for i, t in enumerate(p.discard_tops()):
            if t is not None:
                add(t, 'play_discard', i)
        return acts

    def get_discard_actions(self) -> List[Action]:
        if self.game_over:
            return []
        p = self.current_player
        return [Action('discard', c, None, di)
                for c in p.hand
                for di in range(Player.MAX_DISCARD_PILES)]

    def get_valid_actions(self) -> List[Action]:
        return self.get_play_actions() + self.get_discard_actions()

    # ── Action application ─────────────────────────────────────────────────────
    def apply_action(self, action: Action) -> Tuple[bool, Optional[int]]:
        if self.game_over:
            return False, self.winner
        p = self.current_player

        if action.action_type == 'play_hand':
            if action.card not in p.hand:
                return False, None
            p.remove_from_hand(action.card)
            if not self._apply_to_pile(action.card, action.target_pile, action.wild_value):
                p.add_to_hand(action.card); return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over: self._handle_empty_hand(p)
            return True, self.winner

        elif action.action_type == 'play_stockpile':
            if not p.stockpile_top: return False, None
            card = p.pop_stockpile()
            if not self._apply_to_pile(card, action.target_pile, action.wild_value):
                p.stockpile.append(card); return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over: self._handle_empty_hand(p)
            return True, self.winner

        elif action.action_type == 'play_discard':
            if action.source_pile is None: return False, None
            card = p.pop_discard(action.source_pile)
            if card is None: return False, None
            if not self._apply_to_pile(card, action.target_pile, action.wild_value):
                p.push_discard(card, action.source_pile); return False, None
            self.action_history.append(action)
            self._check_win()
            if not self.game_over: self._handle_empty_hand(p)
            return True, self.winner

        elif action.action_type == 'discard':
            if action.card not in p.hand: return False, None
            p.remove_from_hand(action.card)
            p.push_discard(action.card, action.target_pile)
            self.action_history.append(action)
            self.stats['total_discards'] += 1
            self._stall_ticks = 0   # discarding is productive
            self._switch_turn()
            return True, self.winner

        return False, None

    # ── Win / stall helpers ────────────────────────────────────────────────────
    def _check_win(self):
        if self.current_player.has_won:
            self.game_over = True
            self.winner = self.current_player_idx

    def _handle_empty_hand(self, player: Player):
        """Draw a fresh hand if empty; if deck exhausted, end turn or force winner."""
        if player.hand:
            return
        new_cards = self.deck.draw_many(Player.MAX_HAND_SIZE)
        for c in new_cards:
            player.add_to_hand(c)
        if new_cards:
            self.hand_refills += 1
            return
        # Deck exhausted – player cannot ever discard, so end the turn
        self._stall_ticks += 1
        self._maybe_force_winner()
        if not self.game_over:
            self._switch_turn()

    def _maybe_force_winner(self):
        """
        If both players are stuck (stall_ticks >= 4), end the game.
        The player with the smaller stockpile wins; player 0 wins ties.
        This guarantees there are NEVER any draws.
        """
        if self._stall_ticks >= 4:
            s0 = self.players[0].stockpile_size
            s1 = self.players[1].stockpile_size
            self.winner = 0 if s0 <= s1 else 1
            self.game_over = True

    # ── Unlock convenience accessors ──────────────────────────────────────────
    def is_unlocked(self, player_id: int) -> bool:
        return self.player_unlocked[player_id]

    def aces_needed_to_unlock(self, player_id: int) -> int:
        """How many more Aces does the OPPONENT need to place to unlock player_id?"""
        opp = 1 - player_id
        return max(0, 4 - self.pile_starts_by[opp])

    # ── RL interface ───────────────────────────────────────────────────────────
    def reset(self, seed=None):
        self.__init__(seed if seed is not None else self.seed, self.player_names)
        return self

    def is_terminal(self): return self.game_over

    def get_observation(self) -> Dict[str, Any]:
        p, o = self.current_player, self.opponent
        pid = self.current_player_idx
        return {
            'hand':               [c.base_value for c in p.hand],
            'hand_wild':          [1 if c.is_wild else 0 for c in p.hand],
            'stockpile_top':      p.stockpile_top.base_value if p.stockpile_top else 0,
            'stockpile_top_wild': 1 if (p.stockpile_top and p.stockpile_top.is_wild) else 0,
            'stockpile_size':     p.stockpile_size,
            'discard_tops':       [(c.base_value if c else 0) for c in p.discard_tops()],
            'build_tops':         [self.pile_top_value(i) for i in range(self.MAX_BUILDING_PILES)],
            'opp_stockpile_top':  o.stockpile_top.base_value if o.stockpile_top else 0,
            'opp_stockpile_size': o.stockpile_size,
            'opp_discard_tops':   [(c.base_value if c else 0) for c in o.discard_tops()],
            'opp_hand_size':      o.hand_size,
            'deck_remaining':     self.deck.remaining,
            'turn_number':        self.turn_number,
            'current_player':     pid,
            'phase':              0 if self.phase == self.PHASE_PLAY else 1,
            # Unlock state
            'unlocked':           1 if self.player_unlocked[pid] else 0,
            'opp_unlocked':       1 if self.player_unlocked[1-pid] else 0,
            'my_aces_placed':     self.pile_starts_by[pid],
            'opp_aces_placed':    self.pile_starts_by[1-pid],
        }

    def get_reward(self, player_id: int) -> float:
        if self.game_over:
            return 1.0 if self.winner == player_id else -1.0
        return 0.0

    def copy(self): return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        p, o = self.current_player, self.opponent
        return {
            'turn': self.turn_number,
            'build_tops': [self.pile_top_value(i) for i in range(4)],
            'unlocked': self.player_unlocked[:],
            'pile_starts_by': self.pile_starts_by[:],
            'player': {
                'hand': [c.display() for c in p.hand],
                'stockpile_top': p.stockpile_top.display() if p.stockpile_top else None,
                'stockpile_size': p.stockpile_size,
            },
            'opponent': {
                'hand_size': o.hand_size,
                'stockpile_top': o.stockpile_top.display() if o.stockpile_top else None,
                'stockpile_size': o.stockpile_size,
            },
            'completed_sequences': self.completed_sequences,
            'hand_refills': self.hand_refills,
            'game_over': self.game_over,
            'winner': self.winner,
        }

    def __repr__(self):
        unlocked = ['✓' if u else '✗' for u in self.player_unlocked]
        return (f"GameState(turn={self.turn_number}, "
                f"player={self.current_player.name}, "
                f"unlocked={unlocked})")