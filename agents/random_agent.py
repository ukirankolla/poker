import random
from .base_agent import PokerAgent

class RandomAgent(PokerAgent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def decide(self, context):
        return self.rng.choice(("fold", "call", "raise"))
