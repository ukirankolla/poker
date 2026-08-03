import random
from .card import Card, SUITS, RANKS

class Deck:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.shuffle()

    def shuffle(self):
        self._rng.shuffle(self.cards)

    def draw(self):
        if not self.cards:
            raise RuntimeError("deck is empty")
        return self.cards.pop()

    def draw_many(self, count):
        if count < 0 or count > len(self.cards):
            raise ValueError("not enough cards")
        return [self.draw() for _ in range(count)]
