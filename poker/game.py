from .deck import Deck
from .player import Player
from .evaluator import evaluate, hand_name
from .betting import BettingEngine, BettingPlayer


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

    def _create_betting_engine(self):
        betting_players = [
            BettingPlayer(
                name=player.name,
                stack=player.chips,
            )
            for player in self.players
        ]

        return BettingEngine(
            betting_players,
            minimum_bet=self.big_blind,
        )

    def play_hand(self, verbose=False):
        self.deck = Deck(self.seed)
        self.community = []
        self.pot = 0

        for player in self.players:
            player.reset_for_hand()

        for player in self.players:
            player.hole_cards = self.deck.draw_many(2)

        betting_engine = self._create_betting_engine()

        # Initial MVP betting round:
        # each player checks when no bet is outstanding.
        for betting_player in betting_engine.state.players:
            if not betting_player.folded and not betting_player.all_in:
                betting_engine.check(betting_player)

        self.community = self.deck.draw_many(5)

        active = [player for player in self.players if not player.folded]

        if not active:
            raise ValueError("no active players remain")

        results = [
            (evaluate(player.hole_cards + self.community), player)
            for player in active
        ]

        best = max(score for score, _ in results)
        winners = [player for score, player in results if score == best]

        self.pot = betting_engine.state.pot

        if self.pot == 0:
            self.pot = self.big_blind * len(self.players)

        share = self.pot // len(winners)

        for player in winners:
            player.chips += share

        if verbose:
            print(f"Board: {' '.join(map(str, self.community))}")

            for player in self.players:
                print(
                    f"{player.name}: "
                    f"{' '.join(map(str, player.hole_cards))}"
                )

            print(
                f"Winner: "
                f"{', '.join(player.name for player in winners)} "
                f"({hand_name(best)})"
            )

            print(f"Pot: {self.pot}")

        return winners, best
