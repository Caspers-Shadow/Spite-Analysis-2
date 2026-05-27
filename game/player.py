from typing import List, Optional
from .card import Card


class PlayerState:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []
        self.stockpile: List[Card] = []  # last element is top
        self.discards: List[List[Card]] = [[],[],[],[]]

    def top_stock(self) -> Optional[Card]:
        return self.stockpile[-1] if self.stockpile else None

    def top_discards(self) -> List[Optional[Card]]:
        return [pile[-1] if pile else None for pile in self.discards]
