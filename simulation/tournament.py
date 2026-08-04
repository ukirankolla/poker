"""Multi-table tournament simulation.

Unlike ``simulation.benchmark`` (which replays cash-game style hands
between agents that get rebought), a tournament has a single buy-in,
rising blind levels, bust-out elimination, table balancing, a final
table, heads-up play, and a single champion.
"""

import argparse
import random
from dataclasses import dataclass, field

from poker.game import HoldemGame
from poker.player import Player

DEFAULT_BLIND_LEVELS = (
    (5, 10),
    (10, 20),
    (15, 30),
    (25, 50),
    (50, 100),
    (75, 150),
    (100, 200),
    (150, 300),
    (200, 400),
    (300, 600),
    (500, 1000),
    (800, 1600),
    (1200, 2400),
)


@dataclass
class TournamentConfig:
    starting_chips: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    players_per_table: int = 9
    hands_per_level: int = 20
    blind_levels: tuple = DEFAULT_BLIND_LEVELS
    prize_pool: int | None = None
    seed: int = 42


@dataclass
class TournamentResult:
    standings: list = field(default_factory=list)
    winner: Player | None = None
    hands_played: int = 0
    level: int = 0
    big_blind: int = 0
    payouts: dict = field(default_factory=dict)


@dataclass
class Table:
    players: list = field(default_factory=list)
    button: int = 0


def _alive(table):
    return [player for player in table.players if player.chips > 0]


def _seat(players, per_table, rng):
    shuffled = list(players)
    rng.shuffle(shuffled)

    table_count = max(1, (len(shuffled) + per_table - 1) // per_table)
    tables = [Table() for _ in range(table_count)]

    for index, player in enumerate(shuffled):
        tables[index % table_count].players.append(player)

    return tables


def _rebalance(tables, rng):
    active = [table for table in tables if len(_alive(table)) > 0]

    if len(active) < 2:
        return tables

    def alive_count(table):
        return len(_alive(table))

    biggest = max(active, key=alive_count)
    smallest = min(active, key=alive_count)

    if alive_count(biggest) - alive_count(smallest) <= 1:
        return tables

    if alive_count(biggest) < 3:
        return tables

    candidates = _alive(biggest)
    rng.shuffle(candidates)
    mover = min(candidates, key=lambda player: player.chips)

    biggest.players.remove(mover)
    smallest.players.append(mover)

    return tables


def _merge_into_final_table(tables):
    survivors = [
        player
        for table in tables
        for player in table.players
        if player.chips > 0
    ]

    return [Table(players=survivors)]


def _play_table_hand(
    table,
    table_index,
    hand_index,
    small_blind,
    big_blind,
    seed,
    statistics=None,
):
    players = _alive(table)

    if len(players) < 2:
        return None

    button = table.button % len(players)

    game = HoldemGame(
        players,
        seed=seed + hand_index * 100 + table_index,
        small_blind=small_blind,
        big_blind=big_blind,
        statistics=statistics,
    )
    game.button_index = (button - 1) % len(players)
    winners, score = game.play_hand()

    table.button = button + 1

    return game


def _payouts(config, standings):
    if config.prize_pool is None or not standings:
        return {}

    fractions = {
        1: 0.5,
        2: 0.3,
        3: 0.2,
    }

    payouts = {}

    for position, player in enumerate(standings[::-1], start=1):
        share = fractions.get(position, 0.0)
        payouts[player.name] = round(config.prize_pool * share)

    return payouts


def run_tournament(players, config=None, statistics=None):
    """Run a full tournament and return the result.

    ``players`` are reset to the starting stack and eliminated players
    never come back. Standings list the elimination order, with the
    champion (last remaining player) at the end.

    Pass an existing ``statistics`` tracker to carry opponent profiles
    over from other runs.
    """
    if len(players) < 2:
        raise ValueError("at least two players are required")

    config = config or TournamentConfig()

    if config.hands_per_level < 1:
        raise ValueError("hands_per_level must be at least 1")

    if config.players_per_table < 2:
        raise ValueError("players_per_table must be at least 2")

    rng = random.Random(config.seed)

    for player in players:
        player.chips = config.starting_chips

    tables = _seat(players, config.players_per_table, rng)

    standing = []
    hand_index = 0
    level = 0

    while True:
        level = min(
            hand_index // config.hands_per_level,
            len(config.blind_levels) - 1,
        )
        small_blind, big_blind = config.blind_levels[level]

        for table_index, table in enumerate(tables):
            _play_table_hand(
                table,
                table_index,
                hand_index,
                small_blind,
                big_blind,
                config.seed,
                statistics=statistics,
            )

        hand_index += 1

        for table in tables:
            for player in list(table.players):
                if player.chips <= 0 and player not in standing:
                    standing.append(player)

        alive_count = sum(1 for player in players if player.chips > 0)

        if alive_count <= config.players_per_table:
            tables = _merge_into_final_table(tables)

        if alive_count <= 1:
            break

        tables = _rebalance(tables, rng)

    winner = next(
        (player for player in players if player.chips > 0),
        None,
    )

    if winner is not None and winner not in standing:
        standing.append(winner)

    payouts = _payouts(config, standing)

    return TournamentResult(
        standings=standing,
        winner=winner,
        hands_played=hand_index,
        level=level,
        big_blind=big_blind,
        payouts=payouts,
    )


def print_result(result):
    print("=== Tournament result ===")
    print(f"Hands played: {result.hands_played}")
    print(f"Level reached: {result.level} (big blind {result.big_blind})")

    if result.winner is not None:
        print(f"Champion: {result.winner.name}")

    if result.standings:
        print("\nFinishing order:")
        for position, player in enumerate(result.standings[::-1], start=1):
            payout = result.payouts.get(player.name)
            payout_text = (
                f"  {payout:,} chips" if payout is not None else ""
            )
            print(f"  {position}. {player.name}{payout_text}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a multi-table AI poker tournament."
    )
    parser.add_argument("--players", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument(
        "--agents",
        default="random,rulebased",
        help=(
            "comma-separated list of random, rulebased, ollama; "
            "players are assigned agent types round-robin"
        ),
    )
    parser.add_argument("--hands-per-level", type=int, default=20)
    parser.add_argument("--players-per-table", type=int, default=9)
    parser.add_argument("--prize-pool", type=int, default=None)
    parser.add_argument(
        "--stats-file",
        default=None,
        help=(
            "JSON file that persists per-opponent statistics across runs; "
            "loaded before and saved after the tournament"
        ),
    )
    args = parser.parse_args(argv)

    if args.players < 2:
        parser.error("--players must be at least 2")

    from simulation.benchmark import build_agents

    agent_types = [name.strip() for name in args.agents.split(",")]
    players = []

    for index in range(args.players):
        agent_type = agent_types[index % len(agent_types)]
        agent = build_agents([agent_type], seed=args.seed + index)[0]
        base = type(agent).__name__.replace("Agent", "")
        players.append(Player(f"{base}{index}", agent))

    statistics = None

    if args.stats_file:
        from poker.statistics import StatisticsTracker

        try:
            statistics = StatisticsTracker.load(args.stats_file)
        except (FileNotFoundError, ValueError, TypeError):
            statistics = StatisticsTracker()

    result = run_tournament(
        players,
        TournamentConfig(
            starting_chips=args.chips,
            hands_per_level=args.hands_per_level,
            players_per_table=args.players_per_table,
            prize_pool=args.prize_pool,
            seed=args.seed,
        ),
        statistics=statistics,
    )

    if args.stats_file:
        statistics.save(args.stats_file)

    print_result(result)
    print(
        f"\nTotal chips remaining: "
        f"{sum(player.chips for player in players):,}"
    )


if __name__ == "__main__":
    main()
