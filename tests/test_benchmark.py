import pytest

from agents.ollama_agent import OllamaAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from simulation.benchmark import build_agents, print_report, run_hands


def make_agents(count):
    return [RandomAgent(seed=100 + index) for index in range(count)]


def test_run_hands_is_deterministic():
    first = run_hands(
        [RandomAgent(seed=1), RuleBasedAgent()], hands=50, seed=7
    )
    second = run_hands(
        [RandomAgent(seed=1), RuleBasedAgent()], hands=50, seed=7
    )

    assert first.hands_played == second.hands_played == 50
    assert [p.net_chips for p in first.players] == [
        p.net_chips for p in second.players
    ]
    assert [p.hands_won for p in first.players] == [
        p.hands_won for p in second.players
    ]


def test_run_hands_conserves_chips():
    result = run_hands(make_agents(3), hands=100, seed=1)

    assert result.hands_played == 100
    assert sum(player.net_chips for player in result.players) == 0
    assert sum(player.hands_won for player in result.players) == pytest.approx(
        100
    )
    assert all(0.0 <= player.win_rate <= 1.0 for player in result.players)
    assert all(player.hands == 100 for player in result.players)


def test_run_hands_requires_at_least_two_agents():
    with pytest.raises(ValueError, match="at least two"):
        run_hands([RandomAgent(seed=1)], hands=10)


def test_run_hands_requires_positive_hands():
    with pytest.raises(ValueError, match="at least 1"):
        run_hands(make_agents(2), hands=0)


def test_rebuy_restarts_busted_players():
    result = run_hands(
        [RandomAgent(seed=1), RandomAgent(seed=2)],
        hands=500,
        starting_chips=100,
    )

    assert result.hands_played == 500
    assert result.matches > 1
    assert sum(player.net_chips for player in result.players) == 0


def test_rule_based_only_run_produces_categories():
    result = run_hands(
        [RuleBasedAgent(), RuleBasedAgent(), RuleBasedAgent()],
        hands=100,
        seed=5,
    )

    assert result.hands_played == 100
    assert result.average_pot > 0
    assert sum(result.category_counts.values()) == 100
    assert "Fold" in result.category_counts


def test_ollama_agent_plays_in_benchmark(monkeypatch):
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._probe",
        lambda self: True,
    )
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._query",
        lambda self, context, allowed: (
            "call" if "call" in allowed else allowed[0]
        ),
    )

    result = run_hands(
        [
            RandomAgent(seed=1),
            RuleBasedAgent(),
            OllamaAgent(),
        ],
        hands=20,
        seed=2,
    )

    assert result.hands_played == 20
    assert sum(player.hands_won for player in result.players) == pytest.approx(
        20
    )


def test_print_report_output(capsys):
    result = run_hands(
        [RandomAgent(seed=1), RuleBasedAgent()], hands=30, seed=3
    )

    print_report(result)

    out = capsys.readouterr().out

    assert "benchmark" in out
    assert "Player" in out
    assert "Win%" in out
    assert "EV/hand" in out


def test_build_agents():
    agents = build_agents(["random", "rulebased", "ollama"], seed=1)

    assert len(agents) == 3

    with pytest.raises(ValueError, match="unknown agent"):
        build_agents(["bogus"], seed=1)
