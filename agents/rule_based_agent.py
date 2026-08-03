from .base_agent import PokerAgent
from poker.evaluator import evaluate

class RuleBasedAgent(PokerAgent):
    def decide(self, context):
        cards = list(context.hole_cards) + list(context.community_cards)
        if len(cards) >= 5:
            category = evaluate(cards)[0]
            if category >= 4:
                return "raise"
            if category >= 1:
                return "call"
        return "fold"
