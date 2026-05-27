import random
from typing import List, Optional
from .card import Card, RANKS, SUITS


class Deck:
    def __init__(self, num_decks: int = 2, seed: Optional[int] = None):
        self.num_decks = num_decks
        self.rng = random.Random(seed)
        self.cards: List[Card] = []
        self.build()

    def build(self):
        self.cards = [Card(rank, suit) for _ in range(self.num_decks) for suit in SUITS for rank in RANKS]

    def shuffle(self):
        self.rng.shuffle(self.cards)

    def draw(self, n: int = 1):
        drawn = []
        for _ in range(n):
            if not self.cards:
                break
            drawn.append(self.cards.pop())
        return drawn

    def size(self):
        return len(self.cards)
