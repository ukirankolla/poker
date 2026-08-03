from dataclasses import dataclass, field


@dataclass
class Player:
    name: str
    agent: object
    chips: int = 1000
    folded: bool = False
    current_bet: int = 0
    all_in: bool = False
    hole_cards: list = field(default_factory=list)

    def reset_for_hand(self):
        self.folded = False
        self.current_bet = 0
        self.all_in = False
        self.hole_cards = []
