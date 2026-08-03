import json
import requests
from .base_agent import PokerAgent

class OllamaAgent(PokerAgent):
    def __init__(self, model="qwen3.5:4b", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def decide(self, context):
        prompt = {
            "hole_cards": [str(c) for c in context.hole_cards],
            "community_cards": [str(c) for c in context.community_cards],
            "pot": context.pot,
            "chips": context.chips,
            "allowed_actions": ["fold", "call", "raise"],
            "instruction": "Return JSON only: {"action":"fold|call|raise"}"
        }
        response = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": json.dumps(prompt), "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip().lower()
        for action in ("raise", "call", "fold"):
            if f'"action":"{action}"' in text or f'"action": "{action}"' in text:
                return action
        return "fold"
