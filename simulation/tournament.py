from collections import Counter
from poker.game import HoldemGame

def run_tournament(players, hands=100, seed=42):
    wins = Counter()
    for i in range(hands):
        game = HoldemGame(players, seed=seed + i)
        winners, _ = game.play_hand()
        for winner in winners:
            wins[winner.name] += 1 / len(winners)
    return wins
