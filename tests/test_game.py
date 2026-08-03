from agents.random_agent import RandomAgent
from poker.game import HoldemGame
from poker.player import Player

def test_game_produces_winner():
    players = [Player("A", RandomAgent(1)), Player("B", RandomAgent(2))]
    winners, score = HoldemGame(players, seed=1).play_hand()
    assert winners
    assert score[0] >= 0
