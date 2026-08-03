from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.player import Player
from poker.game import HoldemGame
from simulation.tournament import run_tournament

def main():
    players = [
        Player("RandomBot", RandomAgent(seed=1)),
        Player("RuleBot", RuleBasedAgent()),
        Player("RuleBot2", RuleBasedAgent()),
    ]
    print("=== AI Poker MVP ===")
    game = HoldemGame(players, seed=7)
    game.play_hand(verbose=True)
    print("\n=== Tournament ===")
    for name, wins in run_tournament(players, hands=100).most_common():
        print(f"{name}: {wins:.1f} wins")

if __name__ == "__main__":
    main()
