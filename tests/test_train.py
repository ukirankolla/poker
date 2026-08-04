import json
import random

import pytest

from agents.base_agent import DecisionContext
from agents.learned_agent import LearnedPolicyAgent
from agents.random_agent import RandomAgent
from poker.card import Card
from poker.game import HoldemGame
from poker.player import Player
from simulation.train import (
    Policy,
    read_records,
    train_policy,
)


def synthetic_records(count, seed=1):
    """Records with a learnable decision pattern.

    Fold when the call is more than half the stack, check when it is
    free, raise when the pot is large with no call due, call otherwise.
    """
    rng = random.Random(seed)
    records = []

    for _ in range(count):
        to_call = rng.randint(0, 120)
        chips = rng.randint(20, 1000)
        pot = rng.randint(10, 400)
        street = rng.choice(("preflop", "flop", "turn", "river"))
        position = rng.choice(("button", "small_blind", "big_blind", "middle"))

        if to_call > 0.5 * chips:
            action = "fold"
        elif to_call == 0:
            action = "raise" if pot > 200 else "check"
        else:
            action = "call"

        records.append(
            {
                "player": "S",
                "street": street,
                "position": position,
                "pot": pot,
                "to_call": to_call,
                "chips": chips,
                "current_bet": to_call,
                "minimum_raise": 10,
                "players_remaining": 3,
                "action": action,
                "allowed_actions": ["fold", "check", "call", "raise"],
                "opponent_stats": {},
            }
        )

    return records


def test_train_policy_learns_decision_pattern():
    records = synthetic_records(400)

    policy, metrics = train_policy(records, epochs=60, learning_rate=0.2)

    assert policy.classes
    assert metrics["train_accuracy"] > 0.6
    assert metrics["test_accuracy"] > 0.6
    assert metrics["train_records"] > 0
    assert metrics["test_records"] > 0


def test_train_policy_is_deterministic():
    records = synthetic_records(200, seed=5)

    first, first_metrics = train_policy(records, epochs=30)
    second, second_metrics = train_policy(records, epochs=30)

    assert first.to_dict() == second.to_dict()
    assert first_metrics == second_metrics


def test_policy_save_and_load_round_trip(tmp_path):
    policy, _ = train_policy(synthetic_records(300), epochs=20)

    path = tmp_path / "policy.json"
    policy.save(path)

    restored = Policy.load(path)

    assert restored.to_dict() == policy.to_dict()

    from simulation.train import features_from_record

    record = synthetic_records(1)[0]
    features = features_from_record(record)
    assert restored.predict(features) == policy.predict(features)


def test_policy_predict_returns_trained_actions():
    records = synthetic_records(400)
    policy, _ = train_policy(records, epochs=60)

    probe = records[0]
    prediction = policy.predict(
        __import__("simulation.train", fromlist=["features_from_record"]).features_from_record(
            probe
        )
    )

    assert prediction in policy.classes


def test_read_records_from_jsonl(tmp_path):
    records = synthetic_records(5)

    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    assert read_records(path) == records


def test_train_requires_records():
    with pytest.raises(ValueError, match="at least one"):
        train_policy([])


def test_learned_agent_requires_policy():
    with pytest.raises(ValueError, match="trained policy"):
        LearnedPolicyAgent(None)


def test_learned_agent_plays_in_game():
    policy, _ = train_policy(synthetic_records(400), epochs=60)

    players = [
        Player("Learned", LearnedPolicyAgent(policy), chips=100),
        Player("Random", RandomAgent(seed=1), chips=100),
    ]

    game = HoldemGame(players, seed=2)

    total = sum(player.chips for player in players)
    winners, _ = game.play_hand()

    assert winners
    assert sum(player.chips for player in players) == total


def test_learned_agent_picks_legal_action():
    policy, _ = train_policy(synthetic_records(400), epochs=60)

    agent = LearnedPolicyAgent(policy)
    context = DecisionContext(
        hole_cards=(Card(14, "spades"), Card(14, "hearts")),
        community_cards=(),
        pot=30,
        chips=100,
        current_bet=10,
        player_bet=0,
        minimum_raise=10,
        position="button",
        players_remaining=2,
        allowed_actions=("fold", "call", "all_in"),
    )

    assert agent.decide(context) in {"fold", "call", "all_in"}


def test_train_cli_writes_policy(tmp_path, capsys):
    from simulation.train import main

    data = tmp_path / "data.jsonl"
    output = tmp_path / "policy.json"

    with open(data, "w", encoding="utf-8") as handle:
        for record in synthetic_records(200):
            handle.write(json.dumps(record) + "\n")

    main(["--data", str(data), "--output", str(output), "--epochs", "20"])

    assert output.exists()
    assert "Trained on" in capsys.readouterr().out
