from poker.betting import BettingPlayer, BettingEngine


player1 = BettingPlayer("Alice", 100)
player2 = BettingPlayer("Bob", 100)

engine = BettingEngine(
    [player1, player2],
    minimum_bet=10,
)

print("Initial state:")
print("Alice:", player1.stack)
print("Bob:", player2.stack)
print("Pot:", engine.state.pot)

print("\nAlice bets 10:")
engine.bet(player1, 10)

print("Alice stack:", player1.stack)
print("Alice contribution:", player1.contribution)
print("Current bet:", engine.state.current_bet)
print("Pot:", engine.state.pot)

print("\nBob calls:")
engine.call(player2)

print("Bob stack:", player2.stack)
print("Bob contribution:", player2.contribution)
print("Pot:", engine.state.pot)