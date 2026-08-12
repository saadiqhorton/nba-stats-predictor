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

- **Do not probe or retry `stats.nba.com` from this cloud VM.** Outbound calls hang ~30–60s each (retries multiply that). Player search (`nba_api.stats.static.players`) is offline/static and works; game-log fetches from this AWS egress do not. Treat live NBA game-log fetch as unavailable here — do not curl it, do not run unmocked `fetch_*` against production, and do not wait on a blank Streamlit spinner hoping it recovers.
- **How to verify the app without live NBA data:** run `pytest tests/ --ignore=tests/test_webapp.py` (API is mocked). To demo the UI prediction flow, temporarily patch `src.api.fetch_and_combine_game_logs` with sample DataFrames outside the repo (do not commit that launcher). Do not use `streamlit run app.py` + a real player name as a connectivity test.
- **Why the live site still works:** `www.nbastatmaster.site` runs on a different origin host (behind Cloudflare) whose egress can reach the NBA API. That does not mean this Cursor VM can. Streamlit also caches successful fetches (`@st.cache_data`, TTL 1h).
- **`shap` builds from source** and needs `python3.12-dev` + `build-essential`. The update script only refreshes pip deps; those system packages must already be on the image/snapshot.
- **No project linter is configured** (no ruff/flake8/pre-commit). Use `python -m py_compile` on changed modules or rely on pytest.
- Prefer `./run_app.sh` / `./run_tests.sh` locally; they create `venv` with Python 3.12. In this environment, activate `venv` explicitly before `streamlit` / `pytest`.
- Exclude Playwright e2e by default: `pytest tests/ --ignore=tests/test_webapp.py` (see `README.md` / `tests/README.md` for the Playwright path).
