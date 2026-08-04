import json

from agents.base_agent import DecisionContext
from agents.ollama_agent import OllamaAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.card import Card
from poker.statistics import OpponentStats


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


def test_ollama_agent_falls_back_when_server_unavailable(monkeypatch):
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._probe",
        lambda self: False,
    )

    agent = OllamaAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call"),
    )

    action = agent.decide(context)

    assert action in set(context.allowed_actions)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.ok = True

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_ollama_agent_accepts_valid_json(monkeypatch):
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._probe",
        lambda self: True,
    )
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._query",
        lambda self, context, allowed: "raise",
    )

    agent = OllamaAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call", "raise"),
    )

    action = agent.decide(context)

    assert action == "raise"


def test_ollama_agent_rejects_illegal_action(monkeypatch):
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._probe",
        lambda self: True,
    )
    monkeypatch.setattr(
        "agents.ollama_agent.OllamaAgent._query",
        lambda self, context, allowed: "raise",
    )

    agent = OllamaAgent()
    context = make_context(
        allowed_actions=("fold", "check", "call"),
    )

    action = agent.decide(context)

    assert action in {"fold", "check", "call"}


def test_ollama_agent_sends_structured_chat_request(monkeypatch):
    captured = {}

    class FakeSession:
        def get(self, url, timeout=None):
            captured["probe_url"] = url
            return FakeResponse({"models": []})

        def post(self, url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return FakeResponse(
                {"message": {"content": '{"action": "call"}'}}
            )

    monkeypatch.setattr(
        "agents.ollama_agent.requests.Session",
        lambda: FakeSession(),
    )

    agent = OllamaAgent()
    context = make_context()

    action = agent.decide(context)

    assert action == "call"
    assert captured["probe_url"].endswith("/api/tags")
    assert captured["url"].endswith("/api/chat")

    payload = captured["json"]
    assert payload["model"] == "qwen2.5-coder:1.5b"
    assert payload["messages"][0]["role"] == "system"
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_predict"] == 64


def test_rule_based_agent_raises_with_very_strong_hand():
    agent = RuleBasedAgent(equity_trials=500)
    context = make_context(
        hole_cards=(Card(14, "spades"), Card(14, "hearts")),
        allowed_actions=("fold", "check", "raise", "all_in"),
        players_remaining=2,
    )

    assert agent.decide(context) == "raise"


def test_rule_based_agent_folds_when_equity_below_pot_odds():
    agent = RuleBasedAgent(equity_trials=500)
    context = make_context(
        hole_cards=(Card(7, "clubs"), Card(2, "diamonds")),
        pot=60,
        current_bet=50,
        player_bet=0,
        allowed_actions=("fold", "call"),
        players_remaining=2,
    )

    assert agent.decide(context) == "fold"


def test_rule_based_agent_calls_when_equity_beats_pot_odds():
    agent = RuleBasedAgent(equity_trials=500)
    context = make_context(
        hole_cards=(Card(14, "spades"), Card(13, "spades")),
        community_cards=(
            Card(14, "hearts"),
            Card(3, "clubs"),
            Card(9, "diamonds"),
        ),
        pot=60,
        current_bet=20,
        player_bet=0,
        allowed_actions=("fold", "call"),
        players_remaining=2,
    )

    assert agent.decide(context) == "call"


def test_rule_based_agent_without_equity_stays_passive():
    agent = RuleBasedAgent(equity_trials=0)
    context = make_context(
        hole_cards=(Card(14, "spades"), Card(14, "hearts")),
        allowed_actions=("fold", "check", "raise", "all_in"),
        players_remaining=2,
    )

    assert agent.decide(context) == "check"


def test_ollama_agent_prompt_includes_equity_and_opponent_stats(
    monkeypatch,
):
    captured = {}

    class FakeSession:
        def get(self, url, timeout=None):
            return FakeResponse({"models": []})

        def post(self, url, **kwargs):
            captured["json"] = kwargs.get("json")
            return FakeResponse(
                {"message": {"content": '{"action": "fold"}'}}
            )

    monkeypatch.setattr(
        "agents.ollama_agent.requests.Session",
        lambda: FakeSession(),
    )

    agent = OllamaAgent(equity_trials=100)
    context = make_context(
        hole_cards=(Card(14, "spades"), Card(13, "spades")),
        community_cards=(
            Card(14, "hearts"),
            Card(3, "clubs"),
            Card(9, "diamonds"),
        ),
        pot=80,
        current_bet=20,
        player_bet=0,
        opponent_stats={
            "Opponent": OpponentStats(
                "Opponent",
                hands=10,
                voluntary=6,
            )
        },
    )

    action = agent.decide(context)

    assert action == "fold"

    payload = captured["json"]
    user_content = json.loads(payload["messages"][1]["content"])
    assert "estimated_equity" in user_content
    assert "pot_odds" in user_content
    assert user_content["to_call"] == 20
    assert "opponent_statistics" in user_content
    assert user_content["opponent_statistics"]["Opponent"]["vpip"] == 0.6
