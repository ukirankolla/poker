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
    result = run_tournament(players)
    print(f"Champion: {result.winner.name}")
    print(f"Hands played: {result.hands_played}")

if __name__ == "__main__":
    main()
