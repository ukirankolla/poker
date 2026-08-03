import random

from .base_agent import PokerAgent


class RandomAgent(PokerAgent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def decide(self, context):
        actions = tuple(context.allowed_actions)

        if not actions:
            raise ValueError("no legal actions available")

        return self.rng.choice(actions)
