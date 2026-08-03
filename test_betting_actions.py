from poker.betting import BettingPlayer, BettingEngine


alice = BettingPlayer("Alice", 100)
bob = BettingPlayer("Bob", 100)
charlie = BettingPlayer("Charlie", 100)

engine = BettingEngine(
    [alice, bob, charlie],
    minimum_bet=10,
)

print("=== BET ===")

engine.bet(alice, 10)

print("Alice stack:", alice.stack)
print("Pot:", engine.state.pot)
print("Current bet:", engine.state.current_bet)


print("\n=== CALL ===")

engine.call(bob)

print("Bob stack:", bob.stack)
print("Bob contribution:", bob.contribution)
print("Pot:", engine.state.pot)


print("\n=== RAISE ===")

engine.raise_bet(charlie, 20)

print("Charlie stack:", charlie.stack)
print("Charlie contribution:", charlie.contribution)
print("Current bet:", engine.state.current_bet)
print("Minimum raise:", engine.state.minimum_raise)
print("Pot:", engine.state.pot)


print("\n=== FOLD ===")

engine.fold(alice)

print("Alice folded:", alice.folded)


print("\n=== ALL-IN ===")

engine.all_in(bob)

print("Bob stack:", bob.stack)
print("Bob contribution:", bob.contribution)
print("Bob all-in:", bob.all_in)
print("Pot:", engine.state.pot)