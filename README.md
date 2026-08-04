# AI Poker

A Python Texas Hold'em project designed to experiment with local AI poker agents.

## Current MVP

- Card and deck model
- Five-card hand evaluation
- Texas Hold'em seven-card evaluation
- Player/game model with blinds, side pots, and action validation
- Random agent
- Rule-based agent with Monte Carlo equity estimation and pot-odds play
- Ollama local-LLM agent (equity, pot odds, and opponent stats in the prompt)
- Opponent statistics tracker (VPIP/PFR/3-bet/fold-to-3-bet/aggression)
- Multi-table tournament engine with rising blinds and eliminations
- Benchmark and self-play simulation tooling
- Pytest test suite

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests:

```powershell
pytest
```

Run the MVP:

```powershell
python main.py
```

## Simulations

Head-to-head / multi-agent benchmark (cash-game style with rebuys):

```powershell
python -m simulation.benchmark --hands 200 --agents random,rulebased --show-stats
```

Full tournament with rising blinds, eliminations, and a final table:

```powershell
python -m simulation.tournament --players 20 --seed 42
```

## Local AI with Ollama

Install Ollama and pull a local model:

```powershell
ollama pull qwen2.5-coder:1.5b
```

Make sure Ollama is running, then use `OllamaAgent`. The default model is
`qwen2.5-coder:1.5b`; it returns structured JSON decisions quickly. Reasoning
models (e.g. `qwen3.5*`) ignore the `think:false` override and are not
recommended, because they burn latency on hidden reasoning and can return empty
decisions.

## Roadmap

- [x] Complete legal betting rounds and blinds
- [x] Add action validation and pot accounting
- [x] Add opponent/statistics memory
- [x] Add Monte Carlo equity estimation
- [x] Add stronger local LLM decision prompts
- [x] Add self-play and agent evaluation (benchmark + tournament)
- [ ] Add self-play data pipeline (hand/decision logging for training)
- [ ] Add FastAPI/game UI
- [ ] Add GitHub Actions CI
