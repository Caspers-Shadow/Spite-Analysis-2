from .agent import Agent
from typing import Optional


class HeuristicAgent(Agent):
    """Simple heuristic following priorities:
    1. Play stockpile if possible
    2. Play any card that extends a build (prefer plays from hand that empty or enable stock)
    3. Conserve wilds: prefer non-wild plays
    4. Discard lowest-value card to discard piles
    """
    def __init__(self):
        pass

    def act(self, env, player_index: int):
        actions = env.get_valid_actions(player_index)
        if not actions:
            return None
        # 1. play stockpile
        for a in actions:
            if a["type"] == "PLAY_STOCK":
                return a
        # 2. play hand plays that are not wild (prefer higher value to free future)
        hand_plays = [a for a in actions if a["type"] == "PLAY_HAND"]
        if hand_plays:
            # prefer non-wild plays
            for a in hand_plays:
                hi = a["hand_index"]
                card = env.players[player_index].hand[hi]
                if card.rank != 'K':
                    return a
            return hand_plays[0]
        # 3. play discard top
        for a in actions:
            if a["type"] == "PLAY_DISC":
                return a
        # 4. discard: choose highest index (simple)
        discards = [a for a in actions if a["type"] == "DISCARD"]
        if discards:
            # pick one that discards a non-wild and lowest effective value
            for a in discards:
                hi = a["hand_index"]
                card = env.players[player_index].hand[hi]
                if card.rank != 'K':
                    return a
            return discards[0]
        return actions[0]
