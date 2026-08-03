from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field

from agents.ollama_agent import OllamaAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.evaluator import hand_name
from poker.game import HoldemGame
from poker.player import Player


@dataclass
class PlayerStats:
    name: str
    hands: int = 0
    hands_won: float = 0.0
    net_chips: int = 0
    showdown_wins: int = 0
    fold_wins: int = 0
    net_history: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.hands_won / self.hands if self.hands else 0.0

    @property
    def ev_per_hand(self) -> float:
        return self.net_chips / self.hands if self.hands else 0.0

    @property
    def stddev(self) -> float:
        if len(self.net_history) < 2:
            return 0.0
        return statistics.pstdev(self.net_history)


@dataclass
class BenchmarkResult:
    hands_played: int
    matches: int
    players: list[PlayerStats]
    showdown_hands: int
    average_pot: float
    pot_stddev: float
    category_counts: Counter


def _player_names(agents) -> list[str]:
    seen = Counter()
    names = []

    for agent in agents:
        base = type(agent).__name__.replace("Agent", "")
        seen[base] += 1
        names.append(base if seen[base] == 1 else f"{base}{seen[base]}")

    return names


def run_hands(
    agents,
    hands,
    starting_chips=1000,
    small_blind=5,
    big_blind=10,
    seed=42,
) -> BenchmarkResult:
    """Play ``hands`` hands between ``agents`` and collect statistics.

    Players keep their stacks between hands, so a player who busts
    restarts the next hand with ``starting_chips``; each restart counts
    as a new match. Chips are zero-sum: the players' net chips sum to
    zero over the whole run.
    """
    if hands < 1:
        raise ValueError("hands must be at least 1")

    if len(agents) < 2:
        raise ValueError("at least two agents are required")

    names = _player_names(agents)
    players = [
        Player(name, agent, chips=starting_chips)
        for name, agent in zip(names, agents)
    ]

    game = HoldemGame(
        players,
        seed=seed,
        small_blind=small_blind,
        big_blind=big_blind,
    )

    stats = {player.name: PlayerStats(player.name) for player in players}
    pot_history = []
    category_counts = Counter()
    showdown_hands = 0
    matches = 1

    for _ in range(hands):
        before = {player.name: player.chips for player in players}
        winners, score = game.play_hand()

        is_showdown = score != (9,)

        if is_showdown:
            showdown_hands += 1
            category_counts[hand_name(score)] += 1
        else:
            category_counts["Fold"] += 1

        share = 1.0 / len(winners)

        for winner in winners:
            entry = stats[winner.name]
            entry.hands_won += share
            if is_showdown:
                entry.showdown_wins += 1
            else:
                entry.fold_wins += 1

        for player in players:
            net = player.chips - before[player.name]
            entry = stats[player.name]
            entry.net_chips += net
            entry.net_history.append(net)

        pot_history.append(game.pot)

        if any(player.chips <= 0 for player in players):
            matches += 1
            for player in players:
                player.chips = starting_chips

    for entry in stats.values():
        entry.hands = hands

    return BenchmarkResult(
        hands_played=hands,
        matches=matches,
        players=list(stats.values()),
        showdown_hands=showdown_hands,
        average_pot=statistics.fmean(pot_history),
        pot_stddev=(
            statistics.pstdev(pot_history)
            if len(pot_history) > 1
            else 0.0
        ),
        category_counts=category_counts,
    )


def print_report(result: BenchmarkResult) -> None:
    hands = result.hands_played
    fold_hands = hands - result.showdown_hands

    print(f"=== AI-vs-AI benchmark: {hands:,} hands ===")
    print(
        f"Matches: {result.matches}   "
        f"Showdown: {result.showdown_hands:,} "
        f"({result.showdown_hands / hands:.1%})   "
        f"Fold: {fold_hands:,} ({fold_hands / hands:.1%})"
    )
    print(
        f"Average pot: {result.average_pot:.2f}   "
        f"Pot stddev: {result.pot_stddev:.2f}"
    )
    print()

    header = (
        f"{'Player':<12}{'Hands':>8}{'Won':>8}{'Win%':>8}"
        f"{'Net':>10}{'EV/hand':>10}{'StdDev':>10}"
    )
    print(header)
    print("-" * len(header))

    for player in result.players:
        print(
            f"{player.name:<12}{player.hands:>8,}{player.hands_won:>8.0f}"
            f"{player.win_rate:>8.1%}{player.net_chips:>10,}"
            f"{player.ev_per_hand:>+10.2f}{player.stddev:>10.2f}"
        )

    print()

    if result.category_counts:
        print("Winning hand categories:")
        for name, count in result.category_counts.most_common():
            print(f"  {name:<20}{count:>8,}  {count / hands:>6.1%}")


def build_agents(names, seed=None):
    agents = []

    for index, name in enumerate(names):
        normalized = name.strip().lower()

        if normalized == "random":
            agents.append(
                RandomAgent(
                    seed=None if seed is None else seed + index
                )
            )
        elif normalized == "rulebased":
            agents.append(RuleBasedAgent())
        elif normalized == "ollama":
            agents.append(OllamaAgent())
        else:
            raise ValueError(f"unknown agent: {name!r}")

    return agents


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an AI-vs-AI poker benchmark."
    )
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=5)
    parser.add_argument("--big-blind", type=int, default=10)
    parser.add_argument(
        "--agents",
        default="random,rulebased",
        help=(
            "comma-separated list of random, rulebased, ollama; "
            "ollama requires a running Ollama server, otherwise every "
            "decision waits on its HTTP timeout and the run stalls"
        ),
    )
    args = parser.parse_args(argv)

    agents = build_agents(args.agents.split(","), seed=args.seed)

    start = time.perf_counter()
    result = run_hands(
        agents,
        hands=args.hands,
        starting_chips=args.chips,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start

    print_report(result)
    print(
        f"\nCompleted {result.hands_played:,} hands in {elapsed:.2f}s "
        f"({result.hands_played / elapsed:,.0f} hands/s)"
    )


if __name__ == "__main__":
    main()
