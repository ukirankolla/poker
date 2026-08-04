"""Per-opponent statistics collected while the game runs.

The tracker observes every decision a player makes and folds the
hand-level events into aggregate profiles that agents can read through
``DecisionContext.opponent_stats``.

Statistics:

* ``vpip`` - voluntarily put money in preflop (called/bet/raised) per hand.
* ``pfr`` - preflop raise rate per hand.
* ``three_bet`` - re-raised preflop after a prior raise, given an
  opportunity (a preflop action while already facing a raise).
* ``fold_to_three_bet`` - folded preflop while facing a raise that
  came after a prior raise.
* ``aggression`` - aggressive actions (bet/raise) per passive action
  (check/call).
* ``showdown`` - hands that went to showdown per hand played.
"""

from dataclasses import dataclass

PREFLOP_ACTIONS = {"call", "bet", "raise", "all_in"}


@dataclass
class OpponentStats:
    name: str
    hands: int = 0
    voluntary: int = 0
    preflop_raises: int = 0
    three_bets: int = 0
    three_bet_opportunities: int = 0
    folds_to_three_bet: int = 0
    aggressive: int = 0
    passive: int = 0
    showdowns: int = 0

    @property
    def vpip(self):
        return self.voluntary / self.hands if self.hands else 0.0

    @property
    def pfr(self):
        return self.preflop_raises / self.hands if self.hands else 0.0

    @property
    def three_bet(self):
        if not self.three_bet_opportunities:
            return 0.0

        return self.three_bets / self.three_bet_opportunities

    @property
    def fold_to_three_bet(self):
        if not self.three_bet_opportunities:
            return 0.0

        return self.folds_to_three_bet / self.three_bet_opportunities

    @property
    def aggression(self):
        if not self.passive:
            return float(self.aggressive > 0)

        return self.aggressive / self.passive

    @property
    def showdown(self):
        return self.showdowns / self.hands if self.hands else 0.0


class StatisticsTracker:
    """Collect per-player statistics across a hand or a whole run."""

    def __init__(self):
        self._totals = {}
        self._current = {}

    def record(self, name, action, street, raises_before, to_call_before):
        hand = self._current.setdefault(
            name,
            {
                "voluntary": False,
                "raised": False,
                "three_bet": False,
                "three_bet_opportunity": False,
                "fold_to_three_bet": False,
            },
        )

        stats = self._totals.setdefault(name, OpponentStats(name))

        if action in ("bet", "raise"):
            stats.aggressive += 1
        elif action in ("check", "call"):
            stats.passive += 1

        if street != "preflop":
            return

        if action in PREFLOP_ACTIONS:
            hand["voluntary"] = True

        if action == "raise":
            hand["raised"] = True
            if raises_before >= 1:
                hand["three_bet"] = True

        if raises_before >= 1:
            hand["three_bet_opportunity"] = True

        if (
            action == "fold"
            and to_call_before > 0
            and raises_before >= 1
        ):
            hand["fold_to_three_bet"] = True

    def end_hand(self, showdown, at_showdown=frozenset(), roster=frozenset()):
        for name in roster:
            self._current.setdefault(
                name,
                {
                    "voluntary": False,
                    "raised": False,
                    "three_bet": False,
                    "three_bet_opportunity": False,
                    "fold_to_three_bet": False,
                },
            )

        for name, hand in self._current.items():
            stats = self._totals.setdefault(name, OpponentStats(name))
            stats.hands += 1

            if hand["voluntary"]:
                stats.voluntary += 1
            if hand["raised"]:
                stats.preflop_raises += 1
            if hand["three_bet"]:
                stats.three_bets += 1
            if hand["three_bet_opportunity"]:
                stats.three_bet_opportunities += 1
            if hand["fold_to_three_bet"]:
                stats.folds_to_three_bet += 1
            if showdown and name in at_showdown:
                stats.showdowns += 1

        self._current.clear()

    def snapshot(self, exclude=None):
        result = {}

        for name, stats in self._totals.items():
            if name != exclude:
                result[name] = stats

        return result
