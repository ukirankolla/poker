from __future__ import annotations

from dataclasses import dataclass, field

from poker.betting import BettingEngine, BettingPlayer


@dataclass
class BettingRound:
    engine: BettingEngine
    players: list[BettingPlayer]
    current_index: int = 0
    acted: list[bool] = field(default_factory=list)

    def __post_init__(self):
        if not self.acted:
            self.acted = [False] * len(self.players)

    def current_player(self) -> BettingPlayer | None:
        """Return the next player who can act, without advancing past them."""
        if self.is_complete():
            return None

        for _ in range(len(self.players)):
            player = self.players[self.current_index]

            if not player.folded and not player.all_in:
                return player

            self.current_index = (
                self.current_index + 1
            ) % len(self.players)

        return None

    def next_player(self) -> BettingPlayer | None:
        """Move to and return the next player who can act."""
        self.current_index = (
            self.current_index + 1
        ) % len(self.players)

        return self.current_player()

    def _current(self) -> BettingPlayer:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        for index, candidate in enumerate(self.players):
            if candidate is player:
                self.acted[index] = True
                return player

        raise ValueError("player does not belong to this round")

    def fold(self) -> None:
        player = self._current()
        self.engine.fold(player)
        self.next_player()

    def check(self) -> None:
        player = self._current()
        self.engine.check(player)
        self.next_player()

    def call(self) -> int:
        player = self._current()
        amount = self.engine.call(player)
        self.next_player()

        return amount

    def bet(self, amount: int) -> int:
        player = self._current()
        actual = self.engine.bet(player, amount)
        self.next_player()

        return actual

    def raise_bet(self, amount: int) -> int:
        player = self._current()
        actual = self.engine.raise_bet(player, amount)
        self.next_player()

        return actual

    def all_in(self) -> int:
        player = self._current()
        actual = self.engine.all_in(player)
        self.next_player()

        return actual

    def is_complete(self) -> bool:
        """Determine whether this betting round is finished."""

        active_players = [
            player
            for player in self.players
            if not player.folded
        ]

        # Only one player remains.
        if len(active_players) <= 1:
            return True

        # Players who still need to act.
        actionable_players = [
            (index, player)
            for index, player in enumerate(self.players)
            if not player.folded and not player.all_in
        ]

        # Everyone is all-in.
        if not actionable_players:
            return True

        current_bet = self.engine.state.current_bet

        # Everyone who can act must have acted and matched the bet.
        for index, player in actionable_players:
            if not self.acted[index]:
                return False

            if player.contribution != current_bet:
                return False

        return True
