from .base_agent import PokerAgent
from poker.evaluator import evaluate


class RuleBasedAgent(PokerAgent):
    def decide(self, context):
        cards = list(context.hole_cards) + list(context.community_cards)
        allowed = set(context.allowed_actions)

        if len(cards) >= 5:
            category = evaluate(cards)[0]

            if category >= 4 and "raise" in allowed:
                return "raise"

            if category >= 1:
                if "call" in allowed:
                    return "call"
                if "check" in allowed:
                    return "check"

        if "check" in allowed:
            return "check"

        if "fold" in allowed:
            return "fold"

        if "call" in allowed:
            return "call"

        return next(iter(allowed))
