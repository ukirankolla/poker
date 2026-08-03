from dataclasses import dataclass

@dataclass
class Player:
    name: str
    agent: object
    chips: int = 1000
    folded: bool = False
    current_bet: int = 0

    def reset_for_hand(self):
        self.folded = False
        self.current_bet = 0
