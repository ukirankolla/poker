from agents.random_agent import RandomAgent
from poker.game import HoldemGame
from poker.player import Player


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


def test_game_betting_round_creates_fallback_pot_when_players_check():
    players = [
        Player("A", RandomAgent(1), chips=100),
        Player("B", RandomAgent(2), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert game.pot == 20
    assert len(winners) == 1

    winner = winners[0]
    loser = next(player for player in players if player is not winner)

    assert winner.chips == 120
    assert loser.chips == 100


def test_game_deals_two_hole_cards_and_five_community_cards():
    players = [
        Player("A", RandomAgent(1)),
        Player("B", RandomAgent(2)),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    assert len(players[0].hole_cards) == 2
    assert len(players[1].hole_cards) == 2
    assert len(game.community) == 5
