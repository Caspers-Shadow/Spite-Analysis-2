"""Player state for Spite Analysis."""
from typing import List, Optional
from game.card import Card


class Player:
    """All state owned by a single player."""

    MAX_HAND_SIZE = 5
    MAX_DISCARD_PILES = 4

    def __init__(self, player_id: int, name: str = ""):
        self.player_id = player_id
        self.name = name or f"Player {player_id + 1}"

        self.hand: List[Card] = []
        # stockpile[0] is bottom, stockpile[-1] is top (visible)
        self.stockpile: List[Card] = []
        # Each discard pile is a list; last element is the playable top
        self.discard_piles: List[List[Card]] = [[] for _ in range(self.MAX_DISCARD_PILES)]

    # ── Stockpile ──────────────────────────────────────────────────────────────
    @property
    def stockpile_top(self) -> Optional[Card]:
        return self.stockpile[-1] if self.stockpile else None

    @property
    def stockpile_size(self) -> int:
        return len(self.stockpile)

    @property
    def has_won(self) -> bool:
        return len(self.stockpile) == 0

    def pop_stockpile(self) -> Optional[Card]:
        """Remove and return the top stockpile card."""
        return self.stockpile.pop() if self.stockpile else None

    # ── Hand ───────────────────────────────────────────────────────────────────
    def add_to_hand(self, card: Card):
        self.hand.append(card)

    def remove_from_hand(self, card: Card) -> bool:
        """Remove a specific card from hand. Returns True if found."""
        try:
            self.hand.remove(card)
            return True
        except ValueError:
            return False

    @property
    def hand_size(self) -> int:
        return len(self.hand)

    def cards_to_draw(self) -> int:
        return max(0, self.MAX_HAND_SIZE - len(self.hand))

    # ── Discard piles ─────────────────────────────────────────────────────────
    def discard_tops(self) -> List[Optional[Card]]:
        """Top card of each discard pile (or None for empty piles)."""
        return [pile[-1] if pile else None for pile in self.discard_piles]

    def pop_discard(self, pile_idx: int) -> Optional[Card]:
        """Remove and return the top card from discard pile *pile_idx*."""
        if 0 <= pile_idx < self.MAX_DISCARD_PILES and self.discard_piles[pile_idx]:
            return self.discard_piles[pile_idx].pop()
        return None

    def push_discard(self, card: Card, pile_idx: int) -> bool:
        """Place *card* on discard pile *pile_idx*. Returns False if invalid index."""
        if 0 <= pile_idx < self.MAX_DISCARD_PILES:
            self.discard_piles[pile_idx].append(card)
            return True
        return False

    def discard_pile_size(self, pile_idx: int) -> int:
        return len(self.discard_piles[pile_idx]) if 0 <= pile_idx < self.MAX_DISCARD_PILES else 0

    # ── Deep copy ─────────────────────────────────────────────────────────────
    def copy(self) -> 'Player':
        p = Player(self.player_id, self.name)
        p.hand = [c.copy() for c in self.hand]
        p.stockpile = [c.copy() for c in self.stockpile]
        p.discard_piles = [[c.copy() for c in pile] for pile in self.discard_piles]
        return p

    def __repr__(self):
        return (f"Player({self.name}, hand={self.hand_size}, "
                f"stock={self.stockpile_size})")
