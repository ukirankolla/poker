from dataclasses import dataclass

SUITS = ("clubs", "diamonds", "hearts", "spades")
RANKS = tuple(range(2, 15))
RANK_NAMES = {11: "J", 12: "Q", 13: "K", 14: "A"}

@dataclass(frozen=True, order=True)
class Card:
    rank: int
    suit: str

    def __post_init__(self):
        if self.rank not in RANKS:
            raise ValueError("rank must be between 2 and 14")
        if self.suit not in SUITS:
            raise ValueError(f"invalid suit: {self.suit}")

    def __str__(self):
        return f"{RANK_NAMES.get(self.rank, str(self.rank))}{self.suit[0].upper()}"
