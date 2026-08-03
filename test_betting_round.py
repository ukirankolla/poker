from poker.betting import BettingEngine, BettingPlayer
from poker.betting_round import BettingRound


alice = BettingPlayer("Alice", 100)
bob = BettingPlayer("Bob", 100)
charlie = BettingPlayer("Charlie", 100)

engine = BettingEngine(
    [alice, bob, charlie],
    minimum_bet=10,
)

round_ = BettingRound(
    engine=engine,
    players=[alice, bob, charlie],
)


print("=== INITIAL ===")

print("Current player:", round_.current_player().name)
print("Pot:", engine.state.pot)


print("\n=== ALICE BETS ===")

round_.bet(10)

print("Alice contribution:", alice.contribution)
print("Pot:", engine.state.pot)
print("Next player:", round_.current_player().name)


print("\n=== BOB CALLS ===")

round_.call()

print("Bob contribution:", bob.contribution)
print("Pot:", engine.state.pot)
print("Next player:", round_.current_player().name)


print("\n=== CHARLIE RAISES ===")

round_.raise_bet(20)

print("Charlie contribution:", charlie.contribution)
print("Current bet:", engine.state.current_bet)
print("Minimum raise:", engine.state.minimum_raise)
print("Next player:", round_.current_player().name)


print("\n=== ALICE CALLS ===")

round_.call()

print("Alice contribution:", alice.contribution)
print("Next player:", round_.current_player().name)


print("\n=== BOB CALLS ===")

round_.call()

print("Bob contribution:", bob.contribution)


print("\n=== ROUND STATUS ===")

print("Pot:", engine.state.pot)
print("Current bet:", engine.state.current_bet)
print("Round complete:", round_.is_complete())
print("Current player:", round_.current_player())