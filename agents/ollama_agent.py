import json

import requests

from .base_agent import PokerAgent


class OllamaAgent(PokerAgent):
    """Ask a local Ollama model for poker decisions.

    Game state is sent as a structured JSON prompt and the model
    answers with JSON. The agent reuses an HTTP session, caps the
    response token budget, and caches a short availability probe so
    benchmarks fall back to passive play quickly when the server is
    down instead of waiting on a timeout for every decision.
    """

    def __init__(
        self,
        model="qwen2.5-coder:1.5b",
        host="http://localhost:11434",
        timeout=15,
        max_tokens=64,
        keep_alive="10m",
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self._session = requests.Session()
        self._available = None

    def _probe(self) -> bool:
        """Return True when the Ollama server is reachable."""
        try:
            response = self._session.get(
                f"{self.host}/api/tags",
                timeout=min(2.0, self.timeout),
            )
            return response.ok
        except requests.RequestException:
            return False

    def decide(self, context):
        allowed_actions = tuple(context.allowed_actions)

        if self._available is None:
            self._available = self._probe()

        if self._available:
            action = self._query(context, allowed_actions)

            if action in allowed_actions:
                return action

        if "check" in allowed_actions:
            return "check"

        if "fold" in allowed_actions:
            return "fold"

        if "call" in allowed_actions:
            return "call"

        return allowed_actions[0]

    def _query(self, context, allowed_actions):
        game_state = {
            "hole_cards": [str(card) for card in context.hole_cards],
            "community_cards": [
                str(card) for card in context.community_cards
            ],
            "pot": context.pot,
            "chips": context.chips,
            "current_bet": context.current_bet,
            "player_bet": context.player_bet,
            "minimum_raise": context.minimum_raise,
            "position": context.position,
            "players_remaining": context.players_remaining,
            "allowed_actions": list(allowed_actions),
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Texas Hold'em poker decision engine. "
                    "Choose exactly one action from allowed_actions. "
                    "Return JSON only in this format: "
                    '{"action":"<action>"}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(game_state),
            },
        ]

        try:
            response = self._session.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "keep_alive": self.keep_alive,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            content = response.json()["message"]["content"]
            data = json.loads(content)
            action = str(data.get("action", "")).lower().strip()

            if action in allowed_actions:
                return action

        except (
            requests.RequestException,
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            self._available = False

        return None
