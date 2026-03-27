# reasoning-token-allocation

This repo provides an OpenReward-compatible ORS environment for learning reasoning-token caps on GSM8K tasks.

The objective is to keep accuracy high while minimizing reasoning token spend.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset files

```bash
train-00000-of-00001.parquet
test-00000-of-00001.parquet
```

Place both files in the project root for local runs, or set `ORWD_DATA_DIR` to the directory containing the files.

## Run server

```bash
python3 server.py
```

Server runs on `http://localhost:8080`.

## Docker

```bash
docker build -t reasoning-token-allocation .
docker run --rm -p 8080:8080 -e ORWD_DATA_DIR=/orwd_data -v "$(pwd):/orwd_data" reasoning-token-allocation
```

## Verify ORS endpoints

```bash
curl http://localhost:8080/health
curl http://localhost:8080/list_environments
curl http://localhost:8080/gsm8ktokencapenvironment/splits
curl http://localhost:8080/gsm8ktokencapenvironment/tools
```

## Train policy through ORS client

```bash
python3 train.py --steps 30000 --base-url http://localhost:8080 --env-name gsm8ktokencapenvironment
```

Artifacts are saved to `artifacts/policy.pt`.

## Evaluate learned policy

```bash
python3 evaluate.py --mode policy --checkpoint artifacts/policy.pt --episodes 2000 --base-url http://localhost:8080 --env-name gsm8ktokencapenvironment
```

## Evaluate fixed cap baselines

```bash
python3 evaluate.py --mode fixed --episodes 2000 --base-url http://localhost:8080 --env-name gsm8ktokencapenvironment
```

## ORS behavior

- Splits: `train`, `test`
- Prompt: current GSM8K question
- Tools:
  - `set_token_cap(cap)` chooses a reasoning-token cap
  - `answer(answer)` ends episode and returns reward
- Cap choices: `{128, 512, 1024, 2048, 4096}`
- Reward: `accuracy * (1 - allocated_tokens/4096)`
- `ToolOutput` includes `reward`, `finished`, and `metadata` with `correct`, `cap`, `allocated_tokens`
