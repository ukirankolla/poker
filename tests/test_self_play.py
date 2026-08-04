import json

import pytest

pytestmark = pytest.mark.integration

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from simulation.self_play import (
    SelfPlayConfig,
    collect_decisions,
    record_hands,
    write_jsonl,
)

REQUIRED_KEYS = {
    "player",
    "street",
    "position",
    "action",
    "amount",
    "pot",
    "chips",
    "current_bet",
    "to_call",
    "minimum_raise",
    "players_remaining",
    "hole_cards",
    "community_cards",
    "allowed_actions",
    "opponent_stats",
}


def make_agents():
    return [RandomAgent(seed=1), RuleBasedAgent(equity_trials=0)]


def test_collect_decisions_records_each_decision():
    result = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=10, seed=3),
    )

    assert result.hands_played == 10
    assert result.decisions > 10

    for record in result.records:
        assert REQUIRED_KEYS <= set(record)
        assert isinstance(record["hole_cards"], list)
        assert isinstance(record["community_cards"], list)
        assert record["action"] in record["allowed_actions"]


def test_action_counts_sum_to_decisions():
    result = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=10, seed=3),
    )

    assert sum(result.action_counts.values()) == result.decisions
    assert all(count > 0 for count in result.action_counts.values())


def test_records_include_opponent_statistics():
    result = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=30, seed=5),
    )

    records_with_stats = [
        record
        for record in result.records
        if record["opponent_stats"]
    ]

    assert records_with_stats

    first = records_with_stats[0]["opponent_stats"]
    for player, stats in first.items():
        assert set(stats) == {
            "hands",
            "vpip",
            "pfr",
            "three_bet",
            "fold_to_three_bet",
            "aggression",
            "showdown",
        }


def test_records_are_json_serializable():
    result = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=10, seed=4),
    )

    for record in result.records:
        json.dumps(record)


def test_write_jsonl_round_trip(tmp_path):
    result = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=5, seed=2),
    )

    path = tmp_path / "decisions.jsonl"
    write_jsonl(result.records, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == result.decisions
    assert json.loads(lines[0]) == result.records[0]


def test_record_hands_writes_to_configured_path(tmp_path):
    output = tmp_path / "self_play.jsonl"

    result = record_hands(
        make_agents(),
        SelfPlayConfig(hands=5, seed=1, output_path=str(output)),
    )

    assert output.exists()
    assert len(output.read_text(encoding="utf-8").splitlines()) == (
        result.decisions
    )


def test_requires_at_least_two_agents():
    with pytest.raises(ValueError, match="at least two"):
        collect_decisions([RandomAgent(seed=1)])


def test_requires_positive_hands():
    with pytest.raises(ValueError, match="at least 1"):
        collect_decisions(
            make_agents(),
            SelfPlayConfig(hands=0),
        )


def test_collect_decisions_is_deterministic():
    first = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=10, seed=7),
    )
    second = collect_decisions(
        make_agents(),
        SelfPlayConfig(hands=10, seed=7),
    )

    assert first.records == second.records
