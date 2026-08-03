from __future__ import annotations

from dataclasses import dataclass

from poker.betting import BettingEngine, BettingPlayer


@dataclass
class BettingRound:
    engine: BettingEngine
    players: list[BettingPlayer]
    current_index: int = 0

    def current_player(self) -> BettingPlayer | None:
        """Return the next player who can act."""
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

    def fold(self) -> None:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        self.engine.fold(player)
        self.next_player()

    def check(self) -> None:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        self.engine.check(player)
        self.next_player()

    def call(self) -> int:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        amount = self.engine.call(player)
        self.next_player()

        return amount

    def bet(self, amount: int) -> int:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        actual = self.engine.bet(player, amount)
        self.next_player()

        return actual

    def raise_bet(self, amount: int) -> int:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

        actual = self.engine.raise_bet(player, amount)
        self.next_player()

        return actual

    def all_in(self) -> int:
        player = self.current_player()

        if player is None:
            raise ValueError("betting round is complete")

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
            player
            for player in active_players
            if not player.all_in
        ]

        # Everyone is all-in.
        if not actionable_players:
            return True

        # If there is no bet, we need everyone to act.
        if self.engine.state.current_bet == 0:
            return False

        # Everyone who can act has matched the current bet.
        return all(
            player.contribution == self.engine.state.current_bet
            for player in actionable_players
        )