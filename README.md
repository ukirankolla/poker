# AI Poker

A Python Texas Hold'em project designed to experiment with local AI poker agents:
rule-based play, a local LLM agent, and a self-trained policy that learns from
self-play.

## Current MVP

- Card and deck model
- Five-card hand evaluation
- Texas Hold'em seven-card evaluation
- Player/game model with blinds, side pots, and action validation
- Random agent
- Rule-based agent with Monte Carlo equity estimation and pot-odds play
- Ollama local-LLM agent (street, equity, pot odds, and opponent stats in the
  prompt, with explicit strategy rules)
- Learned agent driven by a softmax policy trained on self-play decisions
- Opponent statistics tracker (VPIP/PFR/3-bet/fold-to-3-bet/aggression) with
  JSON persistence across runs
- Multi-table tournament engine with rising blinds and eliminations
- Self-play decision logging and policy training pipeline
- FastAPI web UI for benchmarks, tournaments, and single-hand play
- Benchmark, self-play, and training simulation tooling
- GitHub Actions Continuous Integration (build, unit/integration/regression tests)
- GitHub Actions Code Coverage + Docker Build pipelines
- GitHub Actions Continuous Deployment (SSH deploy on merge to main)
- Jenkins CI/CD pipeline (alternative, same stages)
- Dockerfile + docker-compose.yml for containerized deployment
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

Persist and reuse opponent statistics across benchmark runs:

```powershell
python -m simulation.benchmark --hands 200 --agents random,rulebased --stats-file stats.json --show-stats
python -m simulation.benchmark --hands 200 --agents random,rulebased --stats-file stats.json --show-stats
```

Full tournament with rising blinds, eliminations, and a final table (also
supports `--stats-file` to carry opponent profiles in):

```powershell
python -m simulation.tournament --players 20 --seed 42
```

## Self-play learning

Log decisions from a self-play session, train a softmax policy, then benchmark
the learned agent:

```powershell
python -m simulation.self_play --hands 200 --agents random,rulebased --output decisions.jsonl
python -m simulation.train decisions.jsonl --epochs 30 --output policy.json
python -m simulation.benchmark --hands 200 --agents learned,random --policy policy.json --show-stats
```

The trained policy is a pure-Python softmax logistic model over normalized
hand-strength, pot-odds, bet, and position features, so no external ML
dependencies are needed.

## Web UI

Start the FastAPI app:

```powershell
uvicorn web.app:app --reload
```

Then open http://127.0.0.1:8000/ for a page that can run benchmarks and
tournaments and play a single hand against an agent. The API also exposes
`POST /api/benchmark`, `POST /api/tournament`, and `POST /api/hand`, returning
400 for unknown agents.

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

## Jenkins CI

A declarative `Jenkinsfile` at the repo root runs the pipeline in a
`python:3.12` Docker container. Each stage turns green with a checkmark on
success:

- **Build Pass** - installs dependencies, byte-compiles all modules, and
  smoke-imports the engine, agents, simulations, and web app
- **Unit Tests** - core engine and agent tests (`pytest -m "not integration
  and not regression"`)
- **Integration Tests** - multi-component tests across agents, benchmarks,
  tournaments, self-play, training, and the web API (`pytest -m integration`)
- **Regression Tests** - guards for previously fixed bugs such as chip
  conservation (`pytest -m regression`)
- **Code Coverage** - full suite under `pytest-cov` (currently ~90%), gated at
  85% line / 85% class / 70% method coverage
- **Publish Reports** - `junit` test-result trend, Cobertura coverage badge,
  and an archived HTML coverage report

Required plugins: JUnit, Cobertura, HTML Publisher, Timestamper. On an
existing Jenkins, create a Pipeline job pointing at this repository (branch
source) and Jenkins picks up the `Jenkinsfile` automatically.

### Continuous Deployment (CD)

The same pipeline includes three CD stages that activate on merges to `main`:

- **Docker Build** - builds a production image tagged with the build number and
  `latest`, using the `Dockerfile` at the repo root
- **Docker Push** - pushes the image to GitHub Container Registry
  (`ghcr.io/ukirankolla/poker`); requires a `ghcr-token` Jenkins credential
- **Deploy** - SSHes to the target host and runs `docker compose pull && docker compose up -d`

## GitHub Actions CI/CD

GitHub Actions is the primary CI/CD for this repository and runs automatically
on every pull request. The `main` branch is protected: **all required checks
below must pass before a PR can merge.**

Four pipelines keep the check labels distinct and professional:

**Continuous Integration** (`.github/workflows/ci.yml`) — every PR + main:

| Check name | What it does |
|---|---|
| `Continuous Integration / Build and Test` | compile + import smoke test, then unit, integration, and regression suites in one job |

**Code Coverage** (`.github/workflows/coverage.yml`) — every PR + main:

| Check name | What it does |
|---|---|
| `Code Coverage / Report` | pytest-cov ~90%, Codecov report, HTML artifact |

**Docker Build** (`.github/workflows/docker.yml`) — every PR + main:

| Check name | What it does |
|---|---|
| `Docker Build / Build Image` | build + validate container; push to GHCR on main |

**Continuous Deployment** (`.github/workflows/cd.yml`) — after CI succeeds on
`main` only:

| Check name | What it does |
|---|---|
| `Continuous Deployment / Deploy to Production` | SSH + `docker compose pull && docker compose up -d` |

**Repository secrets needed for the CD stage:**

| Secret | Purpose |
|---|---|
| `DEPLOY_HOST` | Target server hostname/IP |
| `DEPLOY_USER` | SSH username on the target server |
| `DEPLOY_SSH_KEY` | SSH private key with access to the target server |

On merge to `main`, CI runs the test suites, the Docker pipeline builds,
validates, and pushes the image to GitHub Container Registry as
`ghcr.io/ukirankolla/poker:latest`, then — once CI completes successfully —
the CD pipeline SSHs to the deploy host and runs
`docker compose pull && docker compose up -d`.

## Docker

Build and run locally:

```powershell
docker build -t ai-poker .
docker run -p 8000:8000 ai-poker
```

Or with Docker Compose:

```powershell
docker compose up --build
```

Then open http://127.0.0.1:8000/. The image is ~120 MB (`python:3.12-slim`
base), includes only runtime dependencies, and excludes tests and dev files
via `.dockerignore`.

To pull a pre-built image from GHCR:

```powershell
docker pull ghcr.io/ukirankolla/poker:latest
docker run -p 8000:8000 ghcr.io/ukirankolla/poker:latest
```

## Architecture

```
poker/          game rules, hand evaluation, equity, statistics
agents/         random, rule-based, ollama (LLM), learned (policy)
simulation/     benchmark, tournament, self-play logging, policy training
web/            FastAPI app and HTML page
tests/          pytest suite (game, agents, statistics, simulations, training, web)
```

## Roadmap

- [x] Complete legal betting rounds and blinds
- [x] Add action validation and pot accounting
- [x] Add opponent/statistics memory
- [x] Add Monte Carlo equity estimation
- [x] Add stronger local LLM decision prompts
- [x] Add self-play and agent evaluation (benchmark + tournament)
- [x] Add self-play data pipeline (hand/decision logging for training)
- [x] Add a trained learned agent and persistent opponent profiles
- [x] Add FastAPI/game UI
- [x] Add GitHub Actions CI
