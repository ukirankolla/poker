"""An agent that plays using a policy trained from self-play data.

``simulation.train`` fits a ``Policy`` from the decision records that
``simulation.self_play`` collects. This agent loads that policy and
returns the highest-probability legal action for the current context,
closing the loop: collect data -> train a policy -> play with it.
"""

from .base_agent import PokerAgent
from simulation.train import features_from_context


class LearnedPolicyAgent(PokerAgent):
    def __init__(self, policy):
        if policy is None:
            raise ValueError("a trained policy is required")

        self.policy = policy

    def decide(self, context):
        allowed = set(context.allowed_actions)
        probabilities = self.policy.predict_proba(
            features_from_context(context)
        )

        eligible = [
            (action, probability)
            for action, probability in probabilities.items()
            if action in allowed
        ]

        if not eligible:
            return ("check" if "check" in allowed else "fold")

        return max(eligible, key=lambda pair: pair[1])[0]
