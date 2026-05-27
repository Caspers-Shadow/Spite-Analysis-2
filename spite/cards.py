from dataclasses import dataclass
from typing import List, Optional
import random

RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["H","D","C","S"]

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def is_red_king(self) -> bool:
        return self.rank == "K" and self.suit in ("H","D")

    def is_black_king(self) -> bool:
        return self.rank == "K" and self.suit in ("C","S")

    def rank_value(self) -> Optional[int]:
        """Return numeric value for non-wild ranks: A=1, 2..10, J=11, Q=12, K=13
        Wild kings are handled separately."""
        if self.rank == "A":
            return 1
        if self.rank == "J":
            return 11
        if self.rank == "Q":
            return 12
        if self.rank == "K":
            return 13
        try:
            return int(self.rank)
        except ValueError:
            return None

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


class Deck:
    def __init__(self, num_decks: int = 2, rng: Optional[random.Random] = None):
        self.num_decks = num_decks
        self.rng = rng or random.Random()
        self.cards: List[Card] = []
        self._build()

    def _build(self):
        self.cards = [Card(rank, suit)
                      for _ in range(self.num_decks)
                      for suit in SUITS
                      for rank in RANKS]

    def shuffle(self):
        self.rng.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        drawn = []
        for _ in range(n):
            if not self.cards:
                break
            drawn.append(self.cards.pop())
        return drawn

    def size(self) -> int:
        return len(self.cards)
