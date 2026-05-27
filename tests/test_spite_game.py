import pytest
from spite.game import SpiteGame


def test_setup_and_draw():
    g = SpiteGame(rng_seed=1)
    g.setup()
    # each player has 5 cards in hand and 13 in stockpile
    assert len(g.hands[0]) == 5
    assert len(g.hands[1]) == 5
    assert len(g.stockpiles[0]) == 13
    assert len(g.stockpiles[1]) == 13


def test_turn_progression_smoke():
    g = SpiteGame(rng_seed=2)
    g.setup()
    # run a few turns with random policy until someone wins or 20 turns
    for _ in range(40):
        cp = g.current_player
        g.step_player_turn(cp, lambda game, p: game.simple_random_policy(p))
        if g.game_over:
            assert g.winner in (0,1)
            break
    # no exceptions
