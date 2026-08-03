# AI Poker

A Python Texas Hold'em project designed to experiment with local AI poker agents.

## Current MVP

- Card and deck model
- Five-card hand evaluation
- Texas Hold'em seven-card evaluation
- Player/game model
- Random agent
- Rule-based agent
- Ollama local-LLM agent
- Tournament simulation
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

## Local AI with Ollama

Install Ollama and pull a local model:

```powershell
ollama pull qwen3.5:4b
```

Make sure Ollama is running, then use `OllamaAgent`.

## Roadmap

1. Complete legal betting rounds and blinds
2. Add action validation and pot accounting
3. Add opponent/statistics memory
4. Add Monte Carlo equity estimation
5. Add stronger local LLM decision prompts
6. Add self-play and agent evaluation
7. Add FastAPI/game UI
8. Add GitHub Actions CI
