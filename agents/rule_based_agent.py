from .base_agent import PokerAgent
from poker.card import SUITS
from poker.equity import estimate_equity, pot_odds
from poker.evaluator import evaluate


class RuleBasedAgent(PokerAgent):
    """An equity- and pot-odds-aware rule-based agent.

    Postflop it acts on made hand strength. When Monte Carlo equity
    estimation is enabled (``equity_trials > 0``) it compares its
    equity against the pot odds before calling or folding, and it
    opens aggressively when equity is very high. Estimates are only
    computed when they can change the decision, cached per
    hole/community combination, and seeded from the cards so runs stay
    deterministic.
    """

    def __init__(self, equity_trials=50, seed=None):
        self.equity_trials = equity_trials
        self._base_seed = seed if seed is not None else 0
        self._equity_cache = {}

    def _cards_key(self, context):
        return tuple(context.hole_cards) + tuple(context.community_cards)

    def _seed_for(self, cards):
        seed = self._base_seed

        for card in cards:
            seed = seed * 1009 + card.rank * 37 + SUITS.index(card.suit)

        return seed & 0xFFFFFFFF

    def _equity(self, context):
        if self.equity_trials <= 0:
            return None

        key = self._cards_key(context)

        if key not in self._equity_cache:
            if len(self._equity_cache) > 1024:
                self._equity_cache.clear()

            opponents = max(0, context.players_remaining - 1)

            self._equity_cache[key] = estimate_equity(
                context.hole_cards,
                context.community_cards,
                num_opponents=opponents,
                trials=self.equity_trials,
                seed=self._seed_for(key),
            )

        return self._equity_cache[key]

    def _opponent_aggression_bias(self, context):
        if not context.opponent_stats:
            return 0.0

        seen = [
            stats.aggression
            for stats in context.opponent_stats.values()
            if stats.hands > 0
        ]

        if not seen:
            return 0.0

        average = sum(seen) / len(seen)

        # Aggressive fields push the call threshold up: hands need a
        # little more equity to justify calling a bettor who plays
        # many hands aggressively.
        return min(0.1, max(0.0, average * 0.05))

    def decide(self, context):
        cards = list(context.hole_cards) + list(context.community_cards)
        allowed = set(context.allowed_actions)
        to_call = max(0, context.current_bet - context.player_bet)

        if len(cards) >= 5:
            category = evaluate(cards)[0]

            if category >= 4 and "raise" in allowed:
                return "raise"

            if category >= 1 and to_call == 0:
                return "check"

        needs_equity = to_call > 0 or (
            len(cards) < 5 and "raise" in allowed
        )
        equity = self._equity(context) if needs_equity else None

        if equity is not None and to_call > 0 and "call" in allowed:
            odds = pot_odds(to_call, context.pot)
            margin = equity - odds - self._opponent_aggression_bias(context)

            if margin > 0:
                if margin > 0.35 and "raise" in allowed:
                    return "raise"
                return "call"

            if "fold" in allowed:
                return "fold"

        if equity is not None and equity > 0.75 and "raise" in allowed:
            return "raise"

        if "check" in allowed:
            return "check"

        if "fold" in allowed:
            return "fold"

        if "call" in allowed:
            return "call"

        return next(iter(allowed))
