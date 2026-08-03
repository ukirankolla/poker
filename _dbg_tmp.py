from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.player import Player
from poker.game import HoldemGame

players = [
    Player("RandomBot", RandomAgent(seed=1)),
    Player("RuleBot", RuleBasedAgent()),
    Player("RuleBot2", RuleBasedAgent()),
]

game = HoldemGame(players, seed=7)
game.play_hand(verbose=True)
print("after hand 0:", [p.chips for p in players])

for i in range(100):
    game = HoldemGame(players, seed=42 + i)
    winners, score = game.play_hand()
    chips = [p.chips for p in players]
    print(f"hand {i}: seed={42+i} pot={game.pot} chips={chips} nacts={len(game.action_history)}")
