from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionContext:
    hole_cards: tuple
    community_cards: tuple
    pot: int
    chips: int
    current_bet: int = 0
    player_bet: int = 0
    minimum_raise: int = 0
    position: str = "unknown"
    players_remaining: int = 2
    allowed_actions: tuple = field(
        default_factory=lambda: ("fold", "check", "call", "raise", "all_in")
    )


class PokerAgent(ABC):
    @abstractmethod
    def decide(self, context: DecisionContext) -> str:
        """Return a legal poker action."""
        raise NotImplementedError
