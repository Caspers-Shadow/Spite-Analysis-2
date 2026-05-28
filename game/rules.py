"""
Stateless rule validation helpers for Spite Analysis.

All functions here are pure – they take state objects and return
booleans or lists without mutating anything.
"""

from typing import List, Optional
from game.card import Card
from game.player import Player


def card_can_start_pile(card: Card) -> bool:
    """Return True if the card can begin a new (empty) building pile."""
    # Needs to represent value 1 (Ace)
    return card.can_represent(1)


def card_can_continue_pile(card: Card, pile_top_value: int,
                            wild_hint: Optional[int] = None) -> bool:
    """
    Return True if *card* can be placed on a pile whose current top is
    *pile_top_value* (0 means the pile is empty).

    wild_hint – if given and card is wild, use this as the intended value.
    """
    needed = pile_top_value + 1
    if needed > 12:
        return False   # pile already complete (Queen was last card)
    if card.is_wild:
        wv = wild_hint if wild_hint is not None else needed
        return card.can_represent(wv) and wv == needed
    return card.base_value == needed


def legal_plays_for_card(card: Card, build_tops: List[int]) -> List[int]:
    """
    Given a card and the current top values of up to 4 building piles,
    return the pile indices the card can legally be played on.

    build_tops should be a list of 4 integers (0 = empty pile).
    """
    valid = []
    for idx, top in enumerate(build_tops):
        if card.is_wild:
            needed = top + 1
            if card.can_represent(needed) and 1 <= needed <= 12:
                valid.append(idx)
        else:
            if card.base_value == top + 1 and (top + 1) <= 12:
                valid.append(idx)
    return valid


def is_legal_discard(pile_idx: int) -> bool:
    """Discard pile index 0-3 is always legal."""
    return 0 <= pile_idx < Player.MAX_DISCARD_PILES


def player_has_legal_play(player: Player, build_tops: List[int]) -> bool:
    """Quick check – does the player have ANY legal play this turn?"""
    all_cards: List[Card] = list(player.hand)
    if player.stockpile_top:
        all_cards.append(player.stockpile_top)
    all_cards.extend(c for c in player.discard_tops() if c is not None)

    for card in all_cards:
        if legal_plays_for_card(card, build_tops):
            return True
    return False
