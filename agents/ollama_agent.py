import json
import requests

from .base_agent import PokerAgent


class OllamaAgent(PokerAgent):
    def __init__(
        self,
        model="qwen3.5:4b",
        host="http://localhost:11434",
    ):
        self.model = model
        self.host = host.rstrip("/")

    def decide(self, context):
        allowed_actions = tuple(context.allowed_actions)

        prompt = {
            "hole_cards": [str(card) for card in context.hole_cards],
            "community_cards": [str(card) for card in context.community_cards],
            "pot": context.pot,
            "chips": context.chips,
            "current_bet": context.current_bet,
            "player_bet": context.player_bet,
            "minimum_raise": context.minimum_raise,
            "position": context.position,
            "players_remaining": context.players_remaining,
            "allowed_actions": list(allowed_actions),
            "instruction": (
                "You are a Texas Hold'em poker decision engine. "
                "Choose exactly one action from allowed_actions. "
                "Return JSON only in this format: "
                '{"action":"<action>"}'
            ),
        }

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": json.dumps(prompt),
                    "stream": False,
                    "format": "json",
                },
                timeout=30,
            )
            response.raise_for_status()

            text = response.json().get("response", "").strip()
            data = json.loads(text)
            action = str(data.get("action", "")).lower().strip()

            if action in allowed_actions:
                return action

        except (
            requests.RequestException,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            pass

        if "check" in allowed_actions:
            return "check"

        if "fold" in allowed_actions:
            return "fold"

        if "call" in allowed_actions:
            return "call"

        return allowed_actions[0]
