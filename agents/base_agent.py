from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class DecisionContext:
    hole_cards: tuple
    community_cards: tuple
    pot: int
    chips: int

class PokerAgent(ABC):
    @abstractmethod
    def decide(self, context: DecisionContext) -> str:
        """Return one of fold, call, or raise."""
        raise NotImplementedError
