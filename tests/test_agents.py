from agents.random_agent import RandomAgent
from agents.base_agent import DecisionContext

def test_random_agent_action():
    agent = RandomAgent(seed=1)
    action = agent.decide(DecisionContext((), (), 10, 100))
    assert action in {"fold", "call", "raise"}
