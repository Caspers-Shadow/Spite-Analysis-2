"""Abstract base class for all Spite Analysis agents."""

from abc import ABC, abstractmethod
from typing import Optional
from game.game_state import GameState, Action


class Agent(ABC):
    """
    Base class every AI agent must extend.

    Subclasses implement ``choose_action`` which selects one action
    given the current game state.
    """

    def __init__(self, player_id: int, name: str = "Agent"):
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def choose_action(self, state: GameState) -> Optional[Action]:
        """
        Select and return a legal action for *state*.

        Return None only if there are genuinely no legal actions
        (this should not happen in a well-formed game state).
        """

    def on_game_start(self, state: GameState):
        """Hook called once when a new game begins (optional)."""

    def on_game_end(self, state: GameState, winner_id: Optional[int]):
        """Hook called when the game ends (optional)."""

    def __repr__(self):
        return f"{self.__class__.__name__}(player={self.player_id}, name={self.name!r})"
