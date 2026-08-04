import json

import requests

from .base_agent import PokerAgent
from poker.card import SUITS
from poker.equity import estimate_equity, pot_odds


class OllamaAgent(PokerAgent):
    """Ask a local Ollama model for poker decisions.

    Game state is sent as a structured JSON prompt and the model
    answers with JSON. The agent reuses an HTTP session, caps the
    response token budget, and caches a short availability probe so
    benchmarks fall back to passive play quickly when the server is
    down instead of waiting on a timeout for every decision.

    When Monte Carlo equity estimation is enabled (``equity_trials >
    0``) the prompt also includes estimated equity, pot odds, and
    opponent statistics so the model has the information a real poker
    player would use.
    """

    def __init__(
        self,
        model="qwen2.5-coder:1.5b",
        host="http://localhost:11434",
        timeout=15,
        max_tokens=64,
        keep_alive="10m",
        equity_trials=100,
        seed=None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.equity_trials = equity_trials
        self._base_seed = seed if seed is not None else 0
        self._equity_cache = {}
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

    def _cards_key(self, context):
        return tuple(context.hole_cards) + tuple(context.community_cards)

    def _seed_for(self, cards):
        seed = self._base_seed

        for card in cards:
            seed = seed * 1009 + card.rank * 37 + SUITS.index(card.suit)

        return seed & 0xFFFFFFFF

    def _equity(self, context):
        if self.equity_trials <= 0:
            return None

        key = self._cards_key(context)

        if key not in self._equity_cache:
            if len(self._equity_cache) > 1024:
                self._equity_cache.clear()

            opponents = max(0, context.players_remaining - 1)

            self._equity_cache[key] = estimate_equity(
                context.hole_cards,
                context.community_cards,
                num_opponents=opponents,
                trials=self.equity_trials,
                seed=self._seed_for(key),
            )

        return self._equity_cache[key]

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

    def _game_state(self, context, allowed_actions):
        to_call = max(0, context.current_bet - context.player_bet)

        state = {
            "position": context.position,
            "hole_cards": [str(card) for card in context.hole_cards],
            "community_cards": [
                str(card) for card in context.community_cards
            ],
            "pot": context.pot,
            "to_call": to_call,
            "stack": context.chips,
            "minimum_raise": context.minimum_raise,
            "players_remaining": context.players_remaining,
            "allowed_actions": list(allowed_actions),
        }

        equity = self._equity(context)

        if equity is not None:
            state["estimated_equity"] = round(equity, 4)
            state["pot_odds"] = round(pot_odds(to_call, context.pot), 4)

        if context.opponent_stats:
            state["opponent_statistics"] = {
                name: {
                    "hands": stats.hands,
                    "vpip": round(stats.vpip, 4),
                    "pfr": round(stats.pfr, 4),
                    "three_bet": round(stats.three_bet, 4),
                    "fold_to_three_bet": round(
                        stats.fold_to_three_bet, 4
                    ),
                    "aggression": round(stats.aggression, 4),
                    "showdown": round(stats.showdown, 4),
                }
                for name, stats in context.opponent_stats.items()
            }

        return state

    def _query(self, context, allowed_actions):
        game_state = self._game_state(context, allowed_actions)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Texas Hold'em poker decision engine. "
                    "Choose exactly one action from allowed_actions. "
                    "Return JSON only in this format: "
                    '{"action":"<action>","amount":<chips if raising>}. '
                    "Use estimated_equity and pot_odds to decide: "
                    "call when equity beats pot odds, fold when it "
                    "does not, and raise with strong hands."
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
