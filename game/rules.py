from typing import List, Tuple, Optional
from .card import Card


def required_next_value(build_pile: List[Tuple[Card,int]]) -> int:
    if not build_pile:
        return 1
    return build_pile[-1][1] + 1


def card_can_play_on(build_pile: List[Tuple[Card,int]], card: Card) -> bool:
    req = required_next_value(build_pile)
    if card.rank == 'K':
        # red king can be 1..12, black king 2..12
        if card.is_red_king():
            return 1 <= req <= 12
        return 2 <= req <= 12
    rv = card.rank_value()
    return rv == req


def place_card(build_pile: List[Tuple[Card,int]], card: Card) -> int:
    req = required_next_value(build_pile)
    if card.rank == 'K':
        eff = req
    else:
        eff = card.rank_value()
    build_pile.append((card, eff))
    # if reached Queen (12), clear the pile
    if build_pile[-1][1] == 12:
        build_pile.clear()
    return eff
