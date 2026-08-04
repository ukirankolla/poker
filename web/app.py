"""FastAPI web UI for the poker engine.

Run with::

    uvicorn web.app:app --reload

Then open http://127.0.0.1:8000. The single-page UI lets you run a
benchmark, a tournament, or a single hand between the agents and
inspect the results. All endpoints are JSON so the page is optional.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from poker.game import HoldemGame
from poker.player import Player
from simulation.benchmark import _player_names, build_agents, run_hands
from simulation.tournament import TournamentConfig, run_tournament

app = FastAPI(title="AI Poker", version="1.0.0")


def _build_agents(request_agents, seed, policy_path=None):
    if policy_path is None:
        default = Path("policy.json")
        policy_path = str(default) if default.exists() else None
    try:
        return build_agents(request_agents, seed=seed, policy_path=policy_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


class BenchmarkRequest(BaseModel):
    agents: list[str] = Field(default=["random", "rulebased"])
    hands: int = Field(default=100, ge=1)
    seed: int = 42
    chips: int = Field(default=1000, ge=1)
    policy_path: str | None = None


class TournamentRequest(BaseModel):
    players: int = Field(default=20, ge=2)
    agents: list[str] = Field(default=["random", "rulebased"])
    seed: int = 42
    chips: int = Field(default=1000, ge=1)
    players_per_table: int = Field(default=9, ge=2)
    hands_per_level: int = Field(default=20, ge=1)
    prize_pool: int | None = None
    policy_path: str | None = None


class HandRequest(BaseModel):
    agents: list[str] = Field(default=["random", "rulebased"])
    seed: int = 42
    chips: int = Field(default=100, ge=1)
    policy_path: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE


@app.post("/api/benchmark")
def benchmark(request: BenchmarkRequest):
    agents = _build_agents(
        request.agents, seed=request.seed, policy_path=request.policy_path
    )
    result = run_hands(
        agents,
        hands=request.hands,
        starting_chips=request.chips,
        seed=request.seed,
    )

    return {
        "hands_played": result.hands_played,
        "matches": result.matches,
        "showdown_hands": result.showdown_hands,
        "average_pot": round(result.average_pot, 2),
        "players": [
            {
                "name": player.name,
                "hands": player.hands,
                "hands_won": player.hands_won,
                "win_rate": round(player.win_rate, 4),
                "net_chips": player.net_chips,
                "ev_per_hand": round(player.ev_per_hand, 2),
            }
            for player in result.players
        ],
        "categories": dict(result.category_counts),
    }


@app.post("/api/tournament")
def tournament(request: TournamentRequest):
    agents = _build_agents(
        request.agents, seed=request.seed, policy_path=request.policy_path
    )
    players = [
        Player(
            f"{type(agent).__name__.replace('Agent', '')}{index}",
            agent,
        )
        for index, agent in enumerate(agents)
    ]

    while len(players) < request.players:
        agent = build_agents([request.agents[0]], seed=request.seed)[0]
        base = type(agent).__name__.replace("Agent", "")
        players.append(Player(f"{base}{len(players)}", agent))

    result = run_tournament(
        players,
        TournamentConfig(
            starting_chips=request.chips,
            players_per_table=request.players_per_table,
            hands_per_level=request.hands_per_level,
            prize_pool=request.prize_pool,
            seed=request.seed,
        ),
    )

    return {
        "winner": result.winner.name if result.winner else None,
        "hands_played": result.hands_played,
        "level": result.level,
        "big_blind": result.big_blind,
        "payouts": result.payouts,
        "standings": [
            player.name for player in result.standings[::-1]
        ],
    }


@app.post("/api/hand")
def hand(request: HandRequest):
    agents = _build_agents(
        request.agents, seed=request.seed, policy_path=request.policy_path
    )
    names = _player_names(agents)
    players = [
        Player(name, agent, chips=request.chips)
        for name, agent in zip(names, agents)
    ]

    game = HoldemGame(players, seed=request.seed)
    winners, score = game.play_hand()

    return {
        "board": [str(card) for card in game.community],
        "pot": game.pot,
        "winners": [player.name for player in winners],
        "score": list(score) if score != (9,) else "everyone folded",
        "action_history": game.action_history,
        "players": [
            {
                "name": player.name,
                "chips": player.chips,
                "hole_cards": [str(card) for card in player.hole_cards],
                "folded": player.folded,
            }
            for player in players
        ],
    }


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Poker</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 860px; padding: 0 1rem; color: #1a1a1a; }
  h1 { margin-bottom: 0.25rem; }
  textarea, select, input, button { font: inherit; }
  .row { display: flex; gap: 1rem; align-items: center; margin: 0.5rem 0; flex-wrap: wrap; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
  th, td { border-bottom: 1px solid #eee; padding: 0.4rem; text-align: left; }
  button { cursor: pointer; padding: 0.4rem 0.9rem; border-radius: 6px; border: 1px solid #888; background: #f5f5f5; }
  button:hover { background: #ececec; }
  pre { background: #f7f7f7; padding: 0.75rem; border-radius: 6px; overflow: auto; }
  .muted { color: #777; }
</style>
</head>
<body>
<h1>AI Poker</h1>
<p class="muted">Texas Hold'em with Random, Rule-Based, Ollama, and Learned agents.</p>

<div class="card">
  <div class="row">
    <label>Agents (comma separated): <input id="agents" value="random,rulebased" size="24"></label>
    <label>Hands: <input id="hands" value="100" size="6"></label>
    <label>Seed: <input id="seed" value="42" size="6"></label>
    <button onclick="runBenchmark()">Run benchmark</button>
  </div>
  <pre id="benchmark-output" class="muted">Results will appear here.</pre>
</div>

<div class="card">
  <div class="row">
    <label>Players: <input id="tournament-players" value="20" size="4"></label>
    <button onclick="runTournament()">Run tournament</button>
  </div>
  <pre id="tournament-output" class="muted">Results will appear here.</pre>
</div>

<div class="card">
  <div class="row">
    <button onclick="runHand()">Play one hand</button>
  </div>
  <pre id="hand-output" class="muted">Results will appear here.</pre>
</div>

<script>
async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function renderBenchmark(data) {
  let rows = data.players.map(p =>
    `<tr><td>${p.name}</td><td>${p.hands}</td><td>${p.hands_won}</td>` +
    `<td>${(p.win_rate * 100).toFixed(1)}%</td><td>${p.net_chips}</td>` +
    `<td>${p.ev_per_hand.toFixed(2)}</td></tr>`).join("");
  return `Hands: ${data.hands_played} | Matches: ${data.matches} | Showdowns: ${data.showdown_hands} | Avg pot: ${data.average_pot}\n\n` +
    `<table><tr><th>Player</th><th>Hands</th><th>Won</th><th>Win%</th><th>Net</th><th>EV/hand</th></tr>${rows}</table>`;
}

async function runBenchmark() {
  const out = document.getElementById("benchmark-output");
  out.textContent = "Running...";
  try {
    const data = await post("/api/benchmark", {
      agents: document.getElementById("agents").value.split(",").map(s => s.trim()),
      hands: Number(document.getElementById("hands").value),
      seed: Number(document.getElementById("seed").value),
    });
    out.innerHTML = renderBenchmark(data);
  } catch (error) { out.textContent = "Error: " + error; }
}

async function runTournament() {
  const out = document.getElementById("tournament-output");
  out.textContent = "Running...";
  try {
    const data = await post("/api/tournament", {
      players: Number(document.getElementById("tournament-players").value),
      agents: document.getElementById("agents").value.split(",").map(s => s.trim()),
      seed: Number(document.getElementById("seed").value),
    });
    const standings = data.standings.map((name, i) => `${i + 1}. ${name}`).join("\\n");
    out.innerHTML = `Champion: <b>${data.winner}</b> | Hands: ${data.hands_played} | Level ${data.level} (BB ${data.big_blind})\\n\\nFinishing order:\\n<pre>${standings}</pre>`;
  } catch (error) { out.textContent = "Error: " + error; }
}

async function runHand() {
  const out = document.getElementById("hand-output");
  out.textContent = "Running...";
  try {
    const data = await post("/api/hand", {
      agents: document.getElementById("agents").value.split(",").map(s => s.trim()),
      seed: Number(document.getElementById("seed").value),
    });
    const lines = data.action_history.map(a =>
      `${a.street.padEnd(7)} ${a.player.padEnd(12)} ${a.action}`).join("\\n");
    out.innerHTML = `Board: ${data.board.join(" ")} | Pot: ${data.pot} | Winners: ${data.winners}\\n\\n<pre>${lines}</pre>`;
  } catch (error) { out.textContent = "Error: " + error; }
}
</script>
</body>
</html>
"""
