import pytest

pytestmark = pytest.mark.integration

from fastapi.testclient import TestClient

from web.app import app

client = TestClient(app)


def test_index_serves_ui():
    response = client.get("/")

    assert response.status_code == 200
    assert "AI Poker" in response.text
    assert "Run benchmark" in response.text


def test_benchmark_endpoint():
    response = client.post(
        "/api/benchmark",
        json={"agents": ["random", "rulebased"], "hands": 10, "seed": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["hands_played"] == 10
    assert {player["name"] for player in data["players"]} == {
        "Random",
        "RuleBased",
    }
    assert sum(player["hands_won"] for player in data["players"]) == pytest.approx(
        10
    )


def test_benchmark_endpoint_validates_hands():
    response = client.post(
        "/api/benchmark",
        json={"agents": ["random", "rulebased"], "hands": 0},
    )

    assert response.status_code == 422


def test_tournament_endpoint():
    response = client.post(
        "/api/tournament",
        json={
            "players": 4,
            "agents": ["random", "rulebased"],
            "seed": 2,
            "hands_per_level": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["winner"] is not None
    assert len(data["standings"]) == 4
    assert data["winner"] == data["standings"][0]


def test_hand_endpoint():
    response = client.post(
        "/api/hand",
        json={"agents": ["random", "rulebased"], "seed": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["winners"]) >= 1
    assert data["action_history"]
    assert len(data["board"]) == 5
    assert sum(player["chips"] for player in data["players"]) == 200


def test_hand_endpoint_unknown_agent_errors():
    response = client.post(
        "/api/hand",
        json={"agents": ["bogus"], "seed": 3},
    )

    assert response.status_code == 400
    assert "unknown agent" in response.json()["detail"]
