from .deck import Deck
from .player import Player
from .evaluator import evaluate, hand_name

class HoldemGame:
    def __init__(self, players, seed=None, small_blind=5, big_blind=10):
        if len(players) < 2:
            raise ValueError("at least two players are required")
        self.players = players
        self.seed = seed
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.deck = None
        self.community = []
        self.pot = 0

    def play_hand(self, verbose=False):
        self.deck = Deck(self.seed)
        self.community = []
        self.pot = 0
        for p in self.players:
            p.reset_for_hand()

        for p in self.players:
            p.hole_cards = self.deck.draw_many(2)

        # MVP: deal community cards and evaluate showdown.
        # Betting logic will be expanded in the next iteration.
        self.community = self.deck.draw_many(5)
        active = [p for p in self.players if not p.folded]
        results = [(evaluate(p.hole_cards + self.community), p) for p in active]
        best = max(score for score, _ in results)
        winners = [p for score, p in results if score == best]
        self.pot = self.big_blind * len(self.players)
        share = self.pot // len(winners)
        for p in winners:
            p.chips += share
        if verbose:
            print(f"Board: {' '.join(map(str, self.community))}")
            for p in self.players:
                print(f"{p.name}: {' '.join(map(str, p.hole_cards))}")
            print(f"Winner: {', '.join(p.name for p in winners)} ({hand_name(best)})")
        return winners, best
