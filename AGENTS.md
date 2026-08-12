# AGENTS.md

## Cursor Cloud specific instructions

### Product

NBA Player Stats Predictor — a Streamlit app that fetches NBA game logs, trains per-stat XGBoost models, and predicts next-game PTS/REB/AST. Entry point: `app.py`. See `README.md` for standard install/run/test commands.

### Single required service (local dev)

| Service | Command | Port |
|---------|---------|------|
| Streamlit app | `source venv/bin/activate && streamlit run app.py --server.headless true` | 8501 |

Docker Compose (`docker compose up -d --build`, Nginx on 8088) is optional and only needed for load-balancer work (`docs/features/load-balancer.md`, `tests/test_load_balancer.sh`).

### Non-obvious gotchas

- **`stats.nba.com` is often blocked from cloud/datacenter IPs.** Player search (`nba_api.stats.static.players`) works offline; game-log fetches time out. Unit/integration tests mock the API and do not need network access. For a full UI prediction flow without live NBA data, temporarily patch `src.api.fetch_and_combine_game_logs` with sample DataFrames (do not commit that launcher).
- **`shap` builds from source** and needs `python3.12-dev` + `build-essential`. The update script only refreshes pip deps; those system packages must already be on the image/snapshot.
- **No project linter is configured** (no ruff/flake8/pre-commit). Use `python -m py_compile` on changed modules or rely on pytest.
- Prefer `./run_app.sh` / `./run_tests.sh` locally; they create `venv` with Python 3.12. In this environment, activate `venv` explicitly before `streamlit` / `pytest`.
- Exclude Playwright e2e by default: `pytest tests/ --ignore=tests/test_webapp.py` (see `README.md` / `tests/README.md` for the Playwright path).
