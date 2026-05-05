from __future__ import annotations

import ast
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    REWARD_FILE,
    ALLOW_LOCAL_FALLBACK,
)


class RewardCodeError(ValueError):
    pass


class LLMRewardDesigner:
    """Generate and refine Python reward functions.

    If no API key is configured, this class returns deterministic local
    reward code. That keeps the demo reproducible and avoids token cost while
    developing the system.
    """

    def __init__(
        self,
        reward_file: Path = REWARD_FILE,
        prompts_dir: Optional[Path] = None,
        model: Optional[str] = None,
    ) -> None:
        self.reward_file = reward_file
        self.prompts_dir = prompts_dir or Path(__file__).resolve().parent / "prompts"
        self.provider = self._resolve_provider()
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = GEMINI_MODEL
        elif self.provider == "openrouter":
            self.model = OPENROUTER_MODEL
        else:
            self.model = OPENAI_MODEL

    def generate_initial_reward(self, maze_payload: Dict, instruction: str = "") -> Dict:
        inst = instruction if instruction else maze_payload.get("description", "Design a Python reward function for a 2D Gridworld agent. The agent must reach the goal while avoiding lava and walls.")
        prompt = self._load_prompt("initial_prompt.txt").format(maze=maze_payload, instruction=inst)
        return self._generate(prompt=prompt, mode="initial", metrics=None)

    def refine_reward(self, maze_payload: Dict, metrics: Dict, previous_code: str, instruction: str = "") -> Dict:
        inst = instruction if instruction else maze_payload.get("description", "Design a Python reward function for a 2D Gridworld agent. The agent must reach the goal while avoiding lava and walls.")
        prompt = self._load_prompt("feedback_prompt.txt").format(
            maze=maze_payload,
            metrics=metrics,
            previous_code=previous_code,
            instruction=inst,
        )
        return self._generate(prompt=prompt, mode="refine", metrics=metrics)

    def save_reward_code(self, code: str) -> None:
        code = self.extract_code(code)
        self.validate_reward_code(code)
        self.reward_file.parent.mkdir(parents=True, exist_ok=True)
        self.reward_file.write_text(code.rstrip() + "\n", encoding="utf-8")

    def read_current_reward(self) -> str:
        if not self.reward_file.exists():
            return ""
        return self.reward_file.read_text(encoding="utf-8")

    def _generate(self, prompt: str, mode: str, metrics: Optional[Dict]) -> Dict:
        start = time.perf_counter()
        warning = None
        try:
            if self.provider == "gemini":
                code = self._call_gemini(prompt)
                source = "gemini"
            elif self.provider == "openrouter":
                code = self._call_openrouter(prompt)
                source = "openrouter"
            elif self.provider == "openai":
                code = self._call_openai(prompt)
                source = "openai"
            else:
                code = self._local_reward_code(mode=mode, metrics=metrics)
                source = "local-demo"
            code = self.extract_code(code)
            self.validate_reward_code(code)
        except Exception as exc:
            if not ALLOW_LOCAL_FALLBACK and self.provider != "local":
                raise RuntimeError(f"LLM provider '{self.provider}' failed and local fallback is disabled: {exc}") from exc
            code = self._local_reward_code(mode=mode, metrics=metrics)
            code = self.extract_code(code)
            source = "local-demo-fallback"
            warning = f"LLM provider '{self.provider}' failed, fallback to local reward. Error: {exc}"

        latency = time.perf_counter() - start
        return {
            "code": code,
            "source": source,
            "model": self.model if source in {"openai", "gemini", "openrouter"} else "deterministic-fallback",
            "latency_sec": latency,
            "prompt_preview": prompt[:800],
            "warning": warning,
        }

    @staticmethod
    def _resolve_provider() -> str:
        if LLM_PROVIDER in {"gemini", "openai", "openrouter", "local"}:
            return LLM_PROVIDER
        if OPENROUTER_API_KEY:
            return "openrouter"
        if GEMINI_API_KEY:
            return "gemini"
        if OPENAI_API_KEY:
            return "openai"
        return "local"

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed. Run pip install -r requirements.txt.") from exc

        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)

        max_retries = 4
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You write valid Python reward functions for reinforcement learning. Return code only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if ("429" in err_str or "rate" in err_str.lower()) and "quota" not in err_str.lower():
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                raise
        raise last_exc

    def _call_openrouter(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed. Run pip install -r requirements.txt.") from exc

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

        max_retries = 4
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You write valid Python reward functions for reinforcement learning. Return code only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                raise
        raise last_exc



    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You write valid Python reward functions for reinforcement learning. "
                                "Return only Python code, no Markdown.\n\n"
                                + prompt
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1200,
            },
        }

        max_retries = 4
        last_exc = None
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": GEMINI_API_KEY,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    data = json.loads(response.read().decode("utf-8"))
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    return "".join(part.get("text", "") for part in parts)
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(f"Gemini API returned an unexpected response: {data}") from exc
            except urllib.error.HTTPError as exc:
                last_exc = exc
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 503):
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini API request failed with HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Gemini API request failed: {exc}") from exc

        raise RuntimeError(f"Gemini API failed after {max_retries} retries (last HTTP {last_exc.code})") from last_exc

    @staticmethod
    def extract_code(text: str) -> str:
        fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        return fenced.group(1).strip() if fenced else text.strip()

    @staticmethod
    def validate_reward_code(code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise RewardCodeError(f"Generated reward has invalid Python syntax: {exc}") from exc

        has_function = any(
            isinstance(node, ast.FunctionDef) and node.name == "calculate_reward"
            for node in tree.body
        )
        if not has_function:
            raise RewardCodeError("Generated code must define calculate_reward(state, action, next_state, info=None).")

    @staticmethod
    def _local_reward_code(mode: str, metrics: Optional[Dict]) -> str:
        if mode == "initial":
            return '''def calculate_reward(state, action, next_state, info=None):
    """Initial LLM-style reward with a flawed hazard term."""
    info = info or {}
    if info.get("reached_goal") or next_state.get("success"):
        return 80.0
    if info.get("hit_lava"):
        return 45.0
    if info.get("hit_wall"):
        return -3.0

    old_distance = state.get("shortest_path_distance", state.get("distance_to_goal", 0))
    new_distance = next_state.get("shortest_path_distance", next_state.get("distance_to_goal", old_distance))
    progress = old_distance - new_distance
    nearest_lava = next_state.get("nearest_lava_distance")
    reward = -0.5 + 3.0 * progress
    if nearest_lava is not None and nearest_lava <= 1:
        reward += 8.0
    return float(reward)
'''

        return '''def calculate_reward(state, action, next_state, info=None):
    """Refined reward: verifiable constraints plus dense shaping."""
    info = info or {}
    if info.get("reached_goal") or next_state.get("success"):
        return 120.0
    if info.get("hit_lava"):
        return -140.0
    if info.get("hit_wall"):
        return -12.0
    if info.get("timed_out"):
        return -30.0

    old_distance = state.get("shortest_path_distance", state.get("distance_to_goal", 0))
    new_distance = next_state.get("shortest_path_distance", next_state.get("distance_to_goal", old_distance))
    progress = old_distance - new_distance
    nearest_lava = next_state.get("nearest_lava_distance")

    reward = -1.0 + 6.0 * progress
    if nearest_lava is not None:
        if nearest_lava == 0:
            reward -= 140.0
        elif nearest_lava == 1:
            reward -= 18.0
        elif nearest_lava == 2:
            reward -= 4.0
    return float(reward)
'''

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_dir / filename
        return path.read_text(encoding="utf-8")
