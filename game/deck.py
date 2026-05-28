"""Deck management for Spite Analysis (two standard decks shuffled together)."""
import random
from typing import List, Optional
from game.card import Card, Suit


class Deck:
    """Two standard 52-card decks shuffled together (104 cards total)."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._cards: List[Card] = []
        self._reshuffle_pool: List[Card] = []   # completed pile cards waiting for reshuffle
        self._build()
        self._shuffle()

    # ── Setup ──────────────────────────────────────────────────────────────────
    def _build(self):
        """Populate two full decks."""
        self._cards = []
        for _ in range(2):
            for suit in Suit:
                for rank in Card.RANKS:
                    self._cards.append(Card(rank, suit))

    def _shuffle(self):
        rng = random.Random(self.seed)
        rng.shuffle(self._cards)

    # ── Public interface ───────────────────────────────────────────────────────
    def draw(self) -> Optional[Card]:
        """Draw one card from the top. Reshuffles reshuffle_pool if deck is empty."""
        if not self._cards:
            if self._reshuffle_pool:
                self._cards = self._reshuffle_pool[:]
                self._reshuffle_pool = []
                random.shuffle(self._cards)
            else:
                return None
        return self._cards.pop()

    def draw_many(self, count: int) -> List[Card]:
        """Draw *count* cards; fewer if deck runs dry."""
        result = []
        for _ in range(count):
            c = self.draw()
            if c is None:
                break
            result.append(c)
        return result

    def return_completed_pile(self, cards: List[Card]):
        """Add a completed building pile back so it can be reshuffled later."""
        # Strip wild assignments before returning
        for c in cards:
            stripped = Card(c.rank, c.suit)   # fresh copy, no wild_as
            self._reshuffle_pool.append(stripped)

    @property
    def remaining(self) -> int:
        return len(self._cards)

    @property
    def reshuffle_pool_size(self) -> int:
        return len(self._reshuffle_pool)

    def __len__(self) -> int:
        return len(self._cards)

    def __repr__(self):
        return f"Deck({len(self._cards)} cards, pool={len(self._reshuffle_pool)})"
