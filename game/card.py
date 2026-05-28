"""Card representation for Spite Analysis."""
from enum import Enum
from typing import Optional


class Suit(Enum):
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"

    @property
    def symbol(self) -> str:
        return {'Hearts': '♥', 'Diamonds': '♦', 'Clubs': '♣', 'Spades': '♠'}[self.value]

    @property
    def is_red(self) -> bool:
        return self in (Suit.HEARTS, Suit.DIAMONDS)


class Card:
    """A single playing card in Spite Analysis."""

    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    RANK_VALUES = {r: i + 1 for i, r in enumerate(RANKS)}   # A=1 … K=13
    VALUE_TO_RANK = {v: r for r, v in RANK_VALUES.items()}

    def __init__(self, rank: str, suit: Suit, wild_as: Optional[int] = None):
        self.rank = rank
        self.suit = suit
        self.base_value: int = self.RANK_VALUES[rank]
        self.wild_as: Optional[int] = wild_as   # fixed value when played as wild

    # ── Identity helpers ───────────────────────────────────────────────────────
    @property
    def is_king(self) -> bool:
        return self.rank == 'K'

    @property
    def is_red(self) -> bool:
        return self.suit.is_red

    @property
    def is_wild(self) -> bool:
        return self.is_king

    @property
    def is_red_wild(self) -> bool:
        """Red King → can represent Ace (1) through Queen (12)."""
        return self.is_king and self.is_red

    @property
    def is_black_wild(self) -> bool:
        """Black King → can represent 2 through Queen (12), NOT Ace."""
        return self.is_king and not self.is_red

    # ── Value helpers ──────────────────────────────────────────────────────────
    def can_represent(self, value: int) -> bool:
        """Return True if this card can act as *value* on a building pile."""
        if self.is_red_wild:
            return 1 <= value <= 12
        elif self.is_black_wild:
            return 2 <= value <= 12
        return self.base_value == value

    def effective_value(self) -> int:
        """The value this card currently acts as (wild_as if assigned)."""
        return self.wild_as if self.wild_as is not None else self.base_value

    # ── Utilities ──────────────────────────────────────────────────────────────
    def copy(self) -> 'Card':
        return Card(self.rank, self.suit, self.wild_as)

    def display(self) -> str:
        """Short display string e.g. 'K♥'."""
        return f"{self.rank}{self.suit.symbol}"

    def long_display(self) -> str:
        return f"{self.rank} of {self.suit.value}"

    def __repr__(self):
        s = self.display()
        if self.wild_as is not None:
            s += f"(={self.wild_as})"
        return s

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))
