# Self-Refined RL Reward Designer (CPU-Only)

Reproduction-oriented project for:
**Song et al. (2023) - Self-Refined LLM as Automated Reward Function Designer**. 

This implementation keeps the core loop:

`LLM generate reward code -> RL train/evaluate -> feedback -> LLM refine reward`

but adapts experiments from 3D robotics (Isaac Sim + PPO) to 2D Gridworld (Q-Learning) so it runs on a CPU laptop.

## GitHub Metadata

- **Repository name**: `self-refined-rl-reward-designer-cpu`
- **Short description**: `API-only reproduction of Song et al. (2023): LLM-generated reward design with self-refinement on a CPU-friendly 2D Gridworld (Q-Learning).`

---

## 1) Quick Start

### Install
```powershell
pip install -r requirements.txt
```

### Configure environment
Copy `.env.example` to `.env` and set your provider.

### Run
```powershell
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 2) OpenRouter Setup (recommended)

In `.env`:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-oss-20b:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Notes:
- Free models may be rate-limited.
- If provider fails, system auto-falls back to deterministic local reward (no crash).

---

## 3) Other Providers

### Gemini
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

### OpenAI
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

### Local-only (no API cost)
```env
LLM_PROVIDER=local
```

### Strict paper-like mode (no local fallback)
If you want strict API-only behavior (recommended for final reproduction claims):
```env
ALLOW_LOCAL_FALLBACK=false
```
With this setting, provider failure returns an error immediately.

---

## 4) Key Files

- `app.py`: Flask API + orchestration.
- `rl_core/environment.py`: 2D maze tasks.
- `rl_core/q_learning_agent.py`: Q-learning train/evaluate.
- `llm_module/api_caller.py`: LLM calls + fallback.
- `llm_module/prompts/*.txt`: initial/refine prompts.
- `generated_rewards/current_reward.py`: generated reward function.
- `scripts/start_server.ps1`: local start script for Windows.
- `scripts/list_openrouter.py`, `scripts/list_models.py`: provider utility scripts.

---

## 5) Important API Endpoints

- `GET /api/config`
- `GET /api/tasks`
- `POST /api/generate-reward`
- `POST /api/run-agent`
- `POST /api/self-refine`
- `POST /api/run-all`

---

## 6) Stopping Criteria for Refinement

Refinement stops only when both are satisfied:
- `success_rate >= SUCCESS_THRESHOLD`
- `arpd <= ARPD_THRESHOLD`

Config in `.env`:
```env
SUCCESS_THRESHOLD=0.95
ARPD_THRESHOLD=0.10
MAX_REFINEMENT_ITERATIONS=5
```

---

## 7) If You See HTTP 500 on Reward Generation

Now `/api/generate-reward` returns JSON error details.
Common causes:
- wrong model id,
- key invalid/revoked,
- free-tier quota/rate-limit.

The app now catches provider failures and falls back to `local-demo-fallback`.

---

## 8) Run Tests

```powershell
python -m pytest
```

If `pytest` is missing:
```powershell
pip install pytest
```
