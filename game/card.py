from dataclasses import dataclass

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

    def rank_value(self):
        if self.rank == 'A':
            return 1
        if self.rank == 'J':
            return 11
        if self.rank == 'Q':
            return 12
        if self.rank == 'K':
            return 13
        try:
            return int(self.rank)
        except Exception:
            return None

    def __str__(self):
        return f"{self.rank}{self.suit}"
