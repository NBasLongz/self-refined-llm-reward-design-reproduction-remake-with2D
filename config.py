from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct:free").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
ALLOW_LOCAL_FALLBACK = os.getenv("ALLOW_LOCAL_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}

SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.95"))
ARPD_THRESHOLD = float(os.getenv("ARPD_THRESHOLD", "0.10"))
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "5"))
DEFAULT_EPISODES = int(os.getenv("DEFAULT_EPISODES", "350"))
DEFAULT_EVAL_TRIALS = int(os.getenv("DEFAULT_EVAL_TRIALS", "100"))
INITIAL_EPISODES_FACTOR = float(os.getenv("INITIAL_EPISODES_FACTOR", "0.4"))
EVAL_MULTI_SEEDS = int(os.getenv("EVAL_MULTI_SEEDS", "7"))
STOCHASTIC_EVAL = os.getenv("STOCHASTIC_EVAL", "true").strip().lower() in {"1", "true", "yes", "on"}
EVAL_SLIP_PROB = float(os.getenv("EVAL_SLIP_PROB", "0.06"))
EVAL_RANDOM_START_RADIUS = int(os.getenv("EVAL_RANDOM_START_RADIUS", "1"))

REWARD_FILE = BASE_DIR / "generated_rewards" / "current_reward.py"
MANUAL_REWARD_FILE = BASE_DIR / "generated_rewards" / "manual_reward.py"
HISTORY_FILE = BASE_DIR / "logs" / "run_history.json"
