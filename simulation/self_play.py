"""Self-play data collection.

Plays hands between agents and writes one JSONL record per decision, so
the output can later feed a training or imitation-learning pipeline.
Each record captures the full decision context the agent saw (street,
position, pot, stacks, amount to call, board, hole cards, opponent
statistics) plus the action it chose.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass

from poker.game import HoldemGame
from poker.player import Player
from poker.statistics import StatisticsTracker

STAT_KEYS = (
    "hands",
    "vpip",
    "pfr",
    "three_bet",
    "fold_to_three_bet",
    "aggression",
    "showdown",
)


@dataclass
class SelfPlayConfig:
    hands: int = 500
    starting_chips: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    seed: int = 42
    output_path: str = "self_play.jsonl"


@dataclass
class SelfPlayResult:
    records: list
    hands_played: int
    decisions: int
    action_counts: Counter


def _player_names(agents):
    seen = Counter()
    names = []

    for agent in agents:
        base = type(agent).__name__.replace("Agent", "")
        seen[base] += 1
        names.append(base if seen[base] == 1 else f"{base}{seen[base]}")

    return names


def _build_record(name, street, context, action, amount):
    opponent_stats = {
        player_name: {
            key: getattr(stats, key)
            for key in STAT_KEYS
        }
        for player_name, stats in context.opponent_stats.items()
    }

    return {
        "player": name,
        "street": street,
        "position": context.position,
        "action": action,
        "amount": amount,
        "pot": context.pot,
        "chips": context.chips,
        "current_bet": context.current_bet,
        "to_call": context.current_bet - context.player_bet,
        "minimum_raise": context.minimum_raise,
        "players_remaining": context.players_remaining,
        "hole_cards": [str(card) for card in context.hole_cards],
        "community_cards": [
            str(card) for card in context.community_cards
        ],
        "allowed_actions": list(context.allowed_actions),
        "opponent_stats": opponent_stats,
    }


def collect_decisions(agents, config=None):
    """Play ``hands`` hands and return every decision as a record.

    Agents keep their stacks between hands; a busted player restarts the
    next hand with the starting stack (cash-game style, like the
    benchmark). Returns a ``SelfPlayResult`` holding the JSON-serializable
    records.
    """
    if len(agents) < 2:
        raise ValueError("at least two agents are required")

    config = config or SelfPlayConfig()

    if config.hands < 1:
        raise ValueError("hands must be at least 1")

    names = _player_names(agents)
    players = [
        Player(name, agent, chips=config.starting_chips)
        for name, agent in zip(names, agents)
    ]

    records = []

    def on_decision(name, street, context, action, amount):
        records.append(
            _build_record(name, street, context, action, amount)
        )

    game = HoldemGame(
        players,
        seed=config.seed,
        small_blind=config.small_blind,
        big_blind=config.big_blind,
        statistics=StatisticsTracker(),
        on_decision=on_decision,
    )

    for _ in range(config.hands):
        game.play_hand()

        if any(player.chips <= 0 for player in players):
            for player in players:
                player.chips = config.starting_chips

    return SelfPlayResult(
        records=records,
        hands_played=config.hands,
        decisions=len(records),
        action_counts=Counter(record["action"] for record in records),
    )


def write_jsonl(records, path):
    """Serialize records as JSON lines. Each line stands alone."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def record_hands(agents, config=None):
    """Collect decisions and write them to ``config.output_path``."""
    result = collect_decisions(agents, config)

    config = config or SelfPlayConfig()
    write_jsonl(result.records, config.output_path)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect self-play decision data as JSONL."
    )
    parser.add_argument("--hands", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chips", type=int, default=1000)
    parser.add_argument(
        "--agents",
        default="random,rulebased",
        help=(
            "comma-separated list of random, rulebased, ollama; "
            "ollama queries a local server"
        ),
    )
    parser.add_argument(
        "--output",
        default="self_play.jsonl",
        help="output JSONL file path",
    )
    args = parser.parse_args(argv)

    from simulation.benchmark import build_agents

    agents = build_agents(args.agents.split(","), seed=args.seed)

    config = SelfPlayConfig(
        hands=args.hands,
        starting_chips=args.chips,
        seed=args.seed,
        output_path=args.output,
    )

    result = record_hands(agents, config)

    print(
        f"Collected {result.decisions:,} decisions from "
        f"{result.hands_played:,} hands -> {args.output}"
    )
    print("Action mix: " + ", ".join(
        f"{action} {count:,}" for action, count in result.action_counts.most_common()
    ))


if __name__ == "__main__":
    main()
