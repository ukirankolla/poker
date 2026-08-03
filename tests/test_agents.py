import json

from agents.base_agent import DecisionContext
from agents.ollama_agent import OllamaAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent


def make_context(**kwargs):
    values = {
        "hole_cards": (),
        "community_cards": (),
        "pot": 10,
        "chips": 100,
        "current_bet": 0,
        "player_bet": 0,
        "minimum_raise": 10,
        "position": "button",
        "players_remaining": 2,
        "allowed_actions": ("fold", "check", "call", "raise", "all_in"),
    }
    values.update(kwargs)
    return DecisionContext(**values)


def test_random_agent_returns_allowed_action():
    agent = RandomAgent(seed=1)
    context = make_context()

    action = agent.decide(context)

    assert action in set(context.allowed_actions)


def test_random_agent_respects_restricted_actions():
    agent = RandomAgent(seed=1)
    context = make_context(
        allowed_actions=("fold", "call"),
    )

    action = agent.decide(context)

    assert action in {"fold", "call"}


def test_rule_based_agent_checks_when_no_bet_exists():
    agent = RuleBasedAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call"),
    )

    action = agent.decide(context)

    assert action == "check"


def test_rule_based_agent_folds_when_check_is_not_allowed():
    agent = RuleBasedAgent()
    context = make_context(
        allowed_actions=("fold", "call"),
    )

    action = agent.decide(context)

    assert action == "fold"


def test_ollama_agent_falls_back_when_server_unavailable():
    agent = OllamaAgent(
        model="qwen3.5:4b",
        host="http://localhost:11434",
    )

    context = make_context(
        allowed_actions=("fold", "check", "call"),
    )

    action = agent.decide(context)

    assert action in set(context.allowed_actions)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_ollama_agent_accepts_valid_json(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            {
                "response": json.dumps({"action": "raise"}),
            }
        )

    monkeypatch.setattr(
        "agents.ollama_agent.requests.post",
        fake_post,
    )

    agent = OllamaAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call", "raise"),
    )

    action = agent.decide(context)

    assert action == "raise"


def test_ollama_agent_rejects_illegal_action(monkeypatch):
    def fake_post(*args, **kwargs):
        return FakeResponse(
            {
                "response": json.dumps({"action": "raise"}),
            }
        )

    monkeypatch.setattr(
        "agents.ollama_agent.requests.post",
        fake_post,
    )

    agent = OllamaAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call"),
    )

    action = agent.decide(context)

    assert action in {"fold", "check", "call"}
