from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class BettingPlayer:
    name: str
    stack: int
    folded: bool = False
    all_in: bool = False
    contribution: int = 0
    total_contribution: int = 0

    def commit(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("amount cannot be negative")

        actual = min(amount, self.stack)
        self.stack -= actual
        self.contribution += actual
        self.total_contribution += actual

        if self.stack == 0:
            self.all_in = True

        return actual


@dataclass
class BettingState:
    players: list[BettingPlayer]
    pot: int = 0
    current_bet: int = 0
    minimum_raise: int = 0

    def active_players(self) -> list[BettingPlayer]:
        return [
            player
            for player in self.players
            if not player.folded
        ]

    def to_call(self, player: BettingPlayer) -> int:
        return max(0, self.current_bet - player.contribution)

    def add_to_pot(self, player: BettingPlayer, amount: int) -> int:
        actual = player.commit(amount)
        self.pot += actual
        return actual

    def compute_pots(self) -> list[tuple[int, list[BettingPlayer]]]:
        """Split committed money into the main pot and side pots.

        Contributions are cut at each unique contribution level. Every
        player who put in at least that level contributes to the slice,
        but only non-folded players who did so are eligible to win it.
        A slice that only one player is eligible for is returned as is;
        that player is uncontested for it. Slices are ordered from the
        main pot outward.
        """
        levels = sorted(
            {
                player.total_contribution
                for player in self.players
                if player.total_contribution > 0
            }
        )

        pots = []
        previous = 0

        for level in levels:
            contributors = [
                player
                for player in self.players
                if player.total_contribution >= level
            ]
            eligible = [
                player
                for player in contributors
                if not player.folded
            ]

            amount = (level - previous) * len(contributors)
            pots.append((amount, eligible))
            previous = level

        return pots


class BettingEngine:
    def __init__(
        self,
        players: list[BettingPlayer],
        minimum_bet: int,
    ):
        if not players:
            raise ValueError("at least one player is required")

        if minimum_bet <= 0:
            raise ValueError("minimum_bet must be positive")

        self.minimum_bet = minimum_bet
        self.state = BettingState(
            players=players,
            minimum_raise=minimum_bet,
        )

    def reset_street(self):
        """Reset per-street betting state while preserving the pot.

        Contributions, the current bet, and the minimum raise are
        cleared so the next street starts fresh. The accumulated pot
        is kept.
        """
        for player in self.state.players:
            player.contribution = 0

        self.state.current_bet = 0
        self.state.minimum_raise = self.minimum_bet

    def fold(self, player: BettingPlayer) -> None:
        self._validate_player(player)
        player.folded = True

    def check(self, player: BettingPlayer) -> None:
        self._validate_player(player)

        if self.state.to_call(player) != 0:
            raise ValueError("cannot check when facing a bet")

    def call(self, player: BettingPlayer) -> int:
        self._validate_player(player)

        amount = self.state.to_call(player)

        if amount == 0:
            return 0

        return self.state.add_to_pot(player, amount)

    def bet(self, player: BettingPlayer, amount: int) -> int:
        self._validate_player(player)

        if self.state.current_bet != 0:
            raise ValueError("cannot bet when a bet already exists")

        if amount < self.state.minimum_raise:
            raise ValueError(
                f"bet must be at least {self.state.minimum_raise}"
            )

        actual = self.state.add_to_pot(player, amount)
        self.state.current_bet = player.contribution
        self.state.minimum_raise = actual

        return actual

    def raise_bet(self, player: BettingPlayer, amount: int) -> int:
        self._validate_player(player)

        if self.state.current_bet == 0:
            raise ValueError("cannot raise without an existing bet")

        required = self.state.to_call(player)

        if amount < required + self.state.minimum_raise:
            raise ValueError(
                f"raise must be at least {required + self.state.minimum_raise}"
            )

        actual = self.state.add_to_pot(player, amount)
        previous_bet = self.state.current_bet
        self.state.current_bet = player.contribution

        self.state.minimum_raise = (
            self.state.current_bet - previous_bet
        )

        return actual

    def all_in(self, player: BettingPlayer) -> int:
        self._validate_player(player)

        amount = player.stack

        if amount <= 0:
            player.all_in = True
            return 0

        actual = self.state.add_to_pot(player, amount)

        if player.contribution > self.state.current_bet:
            self.state.minimum_raise = (
                player.contribution - self.state.current_bet
            )
            self.state.current_bet = player.contribution

        return actual

    def _validate_player(self, player: BettingPlayer) -> None:
        if player not in self.state.players:
            raise ValueError("player does not belong to this betting state")

        if player.folded:
            raise ValueError("folded player cannot act")

        if player.all_in:
            raise ValueError("all-in player cannot act")
