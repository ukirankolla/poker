from agents.base_agent import PokerAgent
from agents.random_agent import RandomAgent
from poker.game import HoldemGame
from poker.player import Player


class _CheckCallAgent(PokerAgent):
    """Check whenever possible, otherwise call."""

    def decide(self, context):
        if "check" in context.allowed_actions:
            return "check"
        return "call"


def test_game_produces_winner():
    players = [
        Player("A", RandomAgent(1)),
        Player("B", RandomAgent(2)),
    ]

    winners, score = HoldemGame(players, seed=1).play_hand()

    assert winners
    assert score[0] >= 0


def test_game_initializes_betting_engine():
    players = [
        Player("A", RandomAgent(1), chips=100),
        Player("B", RandomAgent(2), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    engine = game._create_betting_engine()

    assert len(engine.state.players) == 2
    assert engine.state.players[0].name == "A"
    assert engine.state.players[1].name == "B"
    assert engine.state.players[0].stack == 100
    assert engine.state.players[1].stack == 100


def test_game_blinds_create_pot_and_are_awarded():
    players = [
        Player("A", _CheckCallAgent(), chips=100),
        Player("B", _CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert game.pot == 20
    assert len(winners) == 1

    assert sorted(player.chips for player in players) == [90, 110]
    assert sum(player.chips for player in players) == 200


def test_game_deals_two_hole_cards_and_five_community_cards():
    players = [
        Player("A", _CheckCallAgent()),
        Player("B", _CheckCallAgent()),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    assert len(players[0].hole_cards) == 2
    assert len(players[1].hole_cards) == 2
    assert len(game.community) == 5
