from typing import List, Optional, Tuple, Dict
from .cards import Deck, Card
import random


class SpiteGame:
    """Minimal implementation of Spite Analysis environment.

    This implementation focuses on turn structure, legal plays, wild handling,
    and stockpile-based win condition. It is deterministic given RNG seed.
    """

    def __init__(self, rng_seed: Optional[int] = None):
        self.rng = random.Random(rng_seed)
        self.deck = Deck(num_decks=2, rng=self.rng)
        self.deck.shuffle()

        # Two players: 0 and 1
        self.hands: List[List[Card]] = [[], []]
        self.stockpiles: List[List[Card]] = [[], []]
        self.discards: List[List[List[Card]]] = [[[],[],[],[]], [[],[],[],[]]]
        # Building piles: each is a list of (card, effective_value)
        self.builds: List[List[Tuple[Card,int]]] = [[],[],[],[]]

        self.current_player = 0
        self.game_over = False
        self.winner: Optional[int] = None

    def setup(self):
        # Deal stockpiles 13 each
        for p in (0,1):
            cards = self.deck.draw(13)
            # stockpile face-down, we keep last element as top
            self.stockpiles[p] = cards[::-1]
        # Flip top stockpile card visible as last element; we keep structure but
        # allow access to top via helper
        # Draw initial hands
        for p in (0,1):
            self.hands[p] = self.deck.draw(5)

    # --- helpers ---
    def top_stock(self, player: int) -> Optional[Card]:
        return self.stockpiles[player][-1] if self.stockpiles[player] else None

    def top_discards(self, player: int) -> List[Optional[Card]]:
        return [pile[-1] if pile else None for pile in self.discards[player]]

    def top_builds(self) -> List[Optional[int]]:
        return [pile[-1][1] if pile else None for pile in self.builds]

    def can_place_on_build(self, card: Card, build_index: int) -> bool:
        """Check whether a card can be legally placed on a build pile.

        Wild kings are assumed legal if they can represent required value.
        """
        top_val = self.builds[build_index][-1][1] if self.builds[build_index] else 0
        # required next value
        req = top_val + 1

        if card.rank == 'K':
            # red king can be 1..12 (A..Q), black king 2..12
            if card.is_red_king():
                return 1 <= req <= 12
            else:
                return 2 <= req <= 12

        rv = card.rank_value()
        if rv is None:
            return False
        return rv == req

    def place_on_build(self, card: Card, build_index: int) -> int:
        """Place the card on the build pile. Returns the effective numeric value assigned.
        For wild kings, choose the required value.
        """
        top_val = self.builds[build_index][-1][1] if self.builds[build_index] else 0
        req = top_val + 1
        if card.rank == 'K':
            eff = req
        else:
            eff = card.rank_value()
        self.builds[build_index].append((card, eff))
        # if reached Q (12), remove pile
        if self.builds[build_index][-1][1] == 12:
            # completed
            self.builds[build_index] = []
        return eff

    # --- turn flow ---
    def draw_phase(self, player: int):
        while len(self.hands[player]) < 5 and self.deck.size() > 0:
            self.hands[player].extend(self.deck.draw(1))

    def legal_actions(self, player: int) -> List[Dict]:
        """Return list of legal actions in the current state for player.

        Actions are dicts with keys: type, source, card_index (for hand), build_index, discard_index
        """
        actions = []
        # from hand
        for hi, card in enumerate(self.hands[player]):
            for bi in range(len(self.builds)):
                if self.can_place_on_build(card, bi):
                    actions.append({"type":"PLAY_HAND_TO_BUILD","hand_index":hi,"build_index":bi})

        # from stockpile
        ts = self.top_stock(player)
        if ts:
            for bi in range(len(self.builds)):
                if self.can_place_on_build(ts, bi):
                    actions.append({"type":"PLAY_STOCKPILE_TO_BUILD","build_index":bi})

        # from discards
        for di, pile in enumerate(self.discards[player]):
            if pile:
                top = pile[-1]
                for bi in range(len(self.builds)):
                    if self.can_place_on_build(top, bi):
                        actions.append({"type":"PLAY_DISCARD_TO_BUILD","discard_index":di,"build_index":bi})

        # discard actions (must discard exactly one to end turn) - allowed anytime during play phase as a terminal action
        for hi, card in enumerate(self.hands[player]):
            for di in range(4):
                actions.append({"type":"DISCARD_HAND_CARD","hand_index":hi,"discard_index":di})

        # end turn is only legal after discarding; game flow enforces discarding - include as placeholder
        actions.append({"type":"END_TURN"})
        return actions

    def perform_action(self, player: int, action: Dict) -> bool:
        """Perform action, return True if action changed state, False if illegal."""
        if action["type"] == "PLAY_HAND_TO_BUILD":
            hi = action["hand_index"]
            bi = action["build_index"]
            if hi < 0 or hi >= len(self.hands[player]):
                return False
            card = self.hands[player][hi]
            if not self.can_place_on_build(card, bi):
                return False
            self.hands[player].pop(hi)
            self.place_on_build(card, bi)
            # check stockpile empty win
            if not self.stockpiles[player]:
                self.game_over = True
                self.winner = player
            return True

        if action["type"] == "PLAY_STOCKPILE_TO_BUILD":
            bi = action["build_index"]
            ts = self.top_stock(player)
            if not ts or not self.can_place_on_build(ts, bi):
                return False
            self.stockpiles[player].pop()
            self.place_on_build(ts, bi)
            if not self.stockpiles[player]:
                self.game_over = True
                self.winner = player
            return True

        if action["type"] == "PLAY_DISCARD_TO_BUILD":
            di = action["discard_index"]
            bi = action["build_index"]
            pile = self.discards[player][di]
            if not pile:
                return False
            card = pile.pop()
            if not self.can_place_on_build(card, bi):
                # put back
                pile.append(card)
                return False
            self.place_on_build(card, bi)
            return True

        if action["type"] == "DISCARD_HAND_CARD":
            hi = action["hand_index"]
            di = action["discard_index"]
            if hi < 0 or hi >= len(self.hands[player]) or di < 0 or di >= 4:
                return False
            card = self.hands[player].pop(hi)
            self.discards[player][di].append(card)
            # end of turn
            self.current_player = 1 - self.current_player
            return True

        if action["type"] == "END_TURN":
            # allow end turn only by forcing a discard outside; treat as no-op
            self.current_player = 1 - self.current_player
            return True

        return False

    def step_player_turn(self, player: int, policy_fn) -> None:
        """Run a single player's turn using a provided policy function.

        policy_fn(game, player) -> action dict or None to stop playing.
        After policy returns None or performs a discard, turn ends.
        """
        if self.game_over:
            return
        if player != self.current_player:
            raise RuntimeError("Not player's turn")

        # Draw
        self.draw_phase(player)

        # Play phase
        while True:
            action = policy_fn(self, player)
            if action is None:
                break
            ok = self.perform_action(player, action)
            if not ok:
                # illegal action -> stop
                break
            # if action was discard, turn ended
            if action["type"] == "DISCARD_HAND_CARD":
                break

    def simple_random_policy(self, player: int):
        """Example random policy: play any legal non-discard moves at random until none, then discard random hand card."""
        actions = self.legal_actions(player)
        # prefer plays over discards
        play_actions = [a for a in actions if a["type"].startswith("PLAY")]
        if play_actions:
            return self.rng.choice(play_actions)
        # otherwise discard random hand card
        hand = self.hands[player]
        if hand:
            hi = self.rng.randrange(len(hand))
            di = self.rng.randrange(4)
            return {"type":"DISCARD_HAND_CARD","hand_index":hi,"discard_index":di}
        return None
