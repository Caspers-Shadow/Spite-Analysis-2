from typing import List, Dict, Optional, Tuple
from .deck import Deck
from .player import PlayerState
from .rules import card_can_play_on, place_card, required_next_value
from .card import Card
import random


class GameState:
    """RL-style environment for Spite Analysis with 2 players.

    Provides reset(), step(action), get_valid_actions(), get_state(), is_terminal().
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        self.deck = Deck(num_decks=2, seed=seed)
        self.players = [PlayerState("P0"), PlayerState("P1")]
        self.builds: List[List[Tuple[Card,int]]] = [[],[],[],[]]
        self.current_player = 0
        self.done = False
        self.winner: Optional[int] = None
        self.turn_count = 0

    def reset(self):
        self.rng = random.Random(self.seed)
        self.deck = Deck(num_decks=2, seed=self.seed)
        self.deck.shuffle()
        self.builds = [[],[],[],[]]
        self.players = [PlayerState("P0"), PlayerState("P1")]
        # deal stockpiles
        for p in (0,1):
            cards = self.deck.draw(13)
            self.players[p].stockpile = cards[::-1]
        # hands
        for p in (0,1):
            self.players[p].hand = self.deck.draw(5)
        self.current_player = 0
        self.done = False
        self.winner = None
        self.turn_count = 0
        return self.get_state()

    def draw_phase(self, player: int):
        while len(self.players[player].hand) < 5 and self.deck.size() > 0:
            self.players[player].hand.extend(self.deck.draw(1))

    def get_valid_actions(self, player: Optional[int] = None) -> List[Dict]:
        if player is None:
            player = self.current_player
        actions = []
        ps = self.players[player]
        # hand plays
        for hi, card in enumerate(ps.hand):
            for bi in range(4):
                if card_can_play_on(self.builds[bi], card):
                    actions.append({"type":"PLAY_HAND","hand_index":hi,"build_index":bi})
        # stockplay
        ts = ps.top_stock()
        if ts:
            for bi in range(4):
                if card_can_play_on(self.builds[bi], ts):
                    actions.append({"type":"PLAY_STOCK","build_index":bi})
        # discards
        for di, pile in enumerate(ps.discards):
            if pile:
                top = pile[-1]
                for bi in range(4):
                    if card_can_play_on(self.builds[bi], top):
                        actions.append({"type":"PLAY_DISC","discard_index":di,"build_index":bi})
        # discard actions
        for hi, _ in enumerate(ps.hand):
            for di in range(4):
                actions.append({"type":"DISCARD","hand_index":hi,"discard_index":di})
        # end turn (not used; discard ends turn)
        actions.append({"type":"END_TURN"})
        return actions

    def step(self, action: Dict) -> Tuple[Dict,bool,Dict]:
        """Perform an action for current player. Returns (state, done, info)."""
        if self.done:
            return self.get_state(), True, {"reason":"game_over"}
        player = self.current_player
        ps = self.players[player]
        # ensure draw phase
        self.draw_phase(player)
        typ = action.get("type")
        if typ == "PLAY_HAND":
            hi = action["hand_index"]
            bi = action["build_index"]
            if hi < 0 or hi >= len(ps.hand):
                return self.get_state(), False, {"illegal":True}
            card = ps.hand.pop(hi)
            if not card_can_play_on(self.builds[bi], card):
                # revert
                ps.hand.insert(hi, card)
                return self.get_state(), False, {"illegal":True}
            place_card(self.builds[bi], card)
            # check win
            if not ps.stockpile:
                self.done = True
                self.winner = player
            return self.get_state(), self.done, {}

        if typ == "PLAY_STOCK":
            bi = action["build_index"]
            card = ps.stockpile.pop()
            if not card_can_play_on(self.builds[bi], card):
                ps.stockpile.append(card)
                return self.get_state(), False, {"illegal":True}
            place_card(self.builds[bi], card)
            if not ps.stockpile:
                self.done = True
                self.winner = player
            return self.get_state(), self.done, {}

        if typ == "PLAY_DISC":
            di = action["discard_index"]
            bi = action["build_index"]
            pile = ps.discards[di]
            if not pile:
                return self.get_state(), False, {"illegal":True}
            card = pile.pop()
            if not card_can_play_on(self.builds[bi], card):
                pile.append(card)
                return self.get_state(), False, {"illegal":True}
            place_card(self.builds[bi], card)
            return self.get_state(), self.done, {}

        if typ == "DISCARD":
            hi = action["hand_index"]
            di = action["discard_index"]
            if hi < 0 or hi >= len(ps.hand):
                return self.get_state(), False, {"illegal":True}
            card = ps.hand.pop(hi)
            ps.discards[di].append(card)
            # end turn
            self.current_player = 1 - self.current_player
            self.turn_count += 1
            return self.get_state(), self.done, {}

        if typ == "END_TURN":
            self.current_player = 1 - self.current_player
            self.turn_count += 1
            return self.get_state(), self.done, {}

        return self.get_state(), False, {"illegal":True}

    def get_state(self) -> Dict:
        # compact representation
        return {
            "current_player": self.current_player,
            "hands": [[str(c) for c in p.hand] for p in self.players],
            "stock_sizes": [len(p.stockpile) for p in self.players],
            "stock_tops": [str(p.top_stock()) if p.top_stock() else None for p in self.players],
            "discard_tops": [[str(x) if x else None for x in p.top_discards()] for p in self.players],
            "builds": [[(str(c),v) for (c,v) in b] for b in self.builds],
            "turn_count": self.turn_count,
            "done": self.done,
            "winner": self.winner,
        }

    def is_terminal(self) -> bool:
        return self.done
