from __future__ import annotations

import importlib.util
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .environment import GridWorldEnv, make_env


RewardFunction = Callable[[Dict, str, Dict, Optional[Dict]], float]


def fallback_reward(state: Dict, action: str, next_state: Dict, info: Optional[Dict] = None) -> float:
    info = info or {}
    if info.get("reached_goal") or next_state.get("success"):
        return 100.0
    if info.get("hit_lava"):
        return -100.0
    if info.get("hit_wall"):
        return -8.0
    progress = state.get("shortest_path_distance", state.get("distance_to_goal", 0)) - next_state.get(
        "shortest_path_distance", next_state.get("distance_to_goal", 0)
    )
    return float(-1.0 + 4.0 * progress)


def load_reward_function(reward_path: Optional[Path] = None) -> RewardFunction:
    reward_path = reward_path or Path(__file__).resolve().parents[1] / "generated_rewards" / "current_reward.py"
    if not reward_path.exists():
        return fallback_reward

    spec = importlib.util.spec_from_file_location("current_reward_dynamic", reward_path)
    if spec is None or spec.loader is None:
        return fallback_reward

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "calculate_reward", None)
    if not callable(candidate):
        return fallback_reward

    def wrapped(state: Dict, action: str, next_state: Dict, info: Optional[Dict] = None) -> float:
        try:
            return float(candidate(state, action, next_state, info))
        except TypeError:
            return float(candidate(state, action, next_state))

    return wrapped


class QLearningAgent:
    def __init__(
        self,
        env: GridWorldEnv,
        learning_rate: float = 0.18,
        discount_factor: float = 0.94,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.992,
        min_epsilon: float = 0.04,
        seed: int = 42,
    ) -> None:
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table: defaultdict[Tuple[int, int], np.ndarray] = defaultdict(lambda: np.zeros(len(GridWorldEnv.ACTIONS)))

    def choose_action(self, state_key: Tuple[int, int], explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(GridWorldEnv.ACTIONS)))
        return int(np.argmax(self.q_table[state_key]))

    @staticmethod
    def state_key(observation: Dict) -> Tuple[int, int]:
        return tuple(observation["position"])

    def train(
        self,
        reward_fn: RewardFunction,
        episodes: int = 350,
        max_steps: Optional[int] = None,
    ) -> Dict:
        max_steps = max_steps or self.env.max_steps
        episode_returns: List[float] = []
        episode_steps: List[int] = []
        success_flags: List[bool] = []
        lava_hits = 0
        wall_hits = 0
        recent_window = 30
        convergence_episode: Optional[int] = None
        start_time = time.perf_counter()

        for episode in range(episodes):
            obs = self.env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                key = self.state_key(obs)
                action = self.choose_action(key, explore=True)
                action_name = GridWorldEnv.ACTION_NAMES[action]
                result = self.env.step(action)
                next_obs = result.observation
                next_key = self.state_key(next_obs)
                reward = reward_fn(obs, action_name, next_obs, result.info)

                best_next = float(np.max(self.q_table[next_key]))
                old_value = self.q_table[key][action]
                target = reward + self.discount_factor * best_next * (not result.done)
                self.q_table[key][action] = old_value + self.learning_rate * (target - old_value)

                total_reward += reward
                steps += 1
                wall_hits += int(result.info.get("hit_wall", False))
                lava_hits += int(result.info.get("hit_lava", False))
                obs = next_obs

                if result.done:
                    break

            success_flags.append(bool(self.env.success))
            episode_returns.append(float(total_reward))
            episode_steps.append(steps)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode >= recent_window and convergence_episode is None:
                recent_sr = sum(success_flags[-recent_window:]) / recent_window
                if recent_sr >= 0.9:
                    convergence_episode = episode + 1

        cpu_time = time.perf_counter() - start_time
        return {
            "episodes": episodes,
            "episode_returns": episode_returns,
            "episode_steps": episode_steps,
            "training_success_rate": sum(success_flags) / max(1, len(success_flags)),
            "training_lava_hits": lava_hits,
            "training_wall_hits": wall_hits,
            "convergence_episode": convergence_episode,
            "cpu_training_time_sec": cpu_time,
        }

    def evaluate(
        self,
        reward_fn: RewardFunction,
        trials: int = 100,
        max_steps: Optional[int] = None,
        slip_prob: float = 0.0,
        random_start_radius: int = 0,
        seed: Optional[int] = None,
    ) -> Dict:
        max_steps = max_steps or self.env.max_steps
        rng = np.random.default_rng(seed if seed is not None else 0)
        successes = 0
        lava_hits = 0
        wall_hits = 0
        returns: List[float] = []
        steps_list: List[int] = []
        first_trajectory: List[Dict] = []
        start_time = time.perf_counter()

        saved_epsilon = self.epsilon
        self.epsilon = 0.0
        for trial in range(trials):
            obs = self.env.reset()
            if random_start_radius > 0:
                start_pos = _sample_start_position(self.env, rng, random_start_radius)
                self.env.agent_pos = start_pos
                obs = self.env.get_observation()
            total_reward = 0.0
            steps = 0
            trajectory: List[Dict] = [{"x": obs["x"], "y": obs["y"], "event": "start"}]

            for _ in range(max_steps):
                key = self.state_key(obs)
                action = self.choose_action(key, explore=False)
                if slip_prob > 0 and rng.random() < slip_prob:
                    action = int(rng.integers(0, len(GridWorldEnv.ACTIONS)))
                action_name = GridWorldEnv.ACTION_NAMES[action]
                result = self.env.step(action)
                next_obs = result.observation
                reward = reward_fn(obs, action_name, next_obs, result.info)

                total_reward += reward
                steps += 1
                wall_hits += int(result.info.get("hit_wall", False))
                lava_hits += int(result.info.get("hit_lava", False))
                event = "goal" if result.info.get("reached_goal") else "lava" if result.info.get("hit_lava") else "move"
                trajectory.append({"x": next_obs["x"], "y": next_obs["y"], "event": event, "action": action_name})
                obs = next_obs

                if result.done:
                    break

            successes += int(self.env.success)
            returns.append(float(total_reward))
            steps_list.append(steps)
            if trial == 0:
                first_trajectory = trajectory

        self.epsilon = saved_epsilon
        eval_time = time.perf_counter() - start_time
        optimal = self.env.shortest_path_length()
        avg_steps = float(np.mean(steps_list)) if steps_list else math.inf
        arpd = None
        if optimal and optimal > 0:
            arpd = max(0.0, ((avg_steps - optimal) / optimal) * 100.0)

        return {
            "trials": trials,
            "success_rate": successes / max(1, trials),
            "average_reward": float(np.mean(returns)) if returns else 0.0,
            "average_steps": avg_steps,
            "lava_hits": lava_hits,
            "wall_hits": wall_hits,
            "optimal_path_length": optimal,
            "arpd": arpd,
            "cpu_eval_time_sec": eval_time,
            "trajectory": first_trajectory,
        }

    def export_q_table(self) -> Dict[str, List[float]]:
        return {f"{key[0]},{key[1]}": values.round(4).tolist() for key, values in self.q_table.items()}


def run_training_session(
    reward_path: Optional[Path] = None,
    episodes: int = 350,
    eval_trials: int = 100,
    seed: int = 42,
    task_id: str = "lava_maze",
    eval_multi_seeds: int = 1,
    stochastic_eval: bool = False,
    eval_slip_prob: float = 0.0,
    eval_random_start_radius: int = 0,
) -> Dict:
    env = make_env(task_id)
    reward_fn = load_reward_function(reward_path)
    agent = QLearningAgent(env, seed=seed)
    training = agent.train(reward_fn, episodes=episodes)
    hard_tasks = {"lava_lake", "spiral_maze", "trap_room"}
    apply_stochastic = stochastic_eval and task_id in hard_tasks
    slip = eval_slip_prob if apply_stochastic else 0.0
    random_radius = eval_random_start_radius if apply_stochastic else 0

    seed_samples: List[Dict] = []
    seeds = max(1, eval_multi_seeds)
    for i in range(seeds):
        eval_seed = seed + 1000 + i
        seed_eval = agent.evaluate(
            reward_fn,
            trials=eval_trials,
            slip_prob=slip,
            random_start_radius=random_radius,
            seed=eval_seed,
        )
        seed_samples.append(seed_eval)

    def _mean(key: str) -> float:
        vals = [float(s[key]) for s in seed_samples if s.get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key: str) -> float:
        vals = [float(s[key]) for s in seed_samples if s.get(key) is not None]
        return float(np.std(vals)) if vals else 0.0

    evaluation = dict(seed_samples[0])
    evaluation["success_rate"] = _mean("success_rate")
    evaluation["average_reward"] = _mean("average_reward")
    evaluation["average_steps"] = _mean("average_steps")
    evaluation["lava_hits"] = _mean("lava_hits")
    evaluation["wall_hits"] = _mean("wall_hits")
    arpd_vals = [float(s["arpd"]) for s in seed_samples if s.get("arpd") is not None]
    evaluation["arpd"] = float(np.mean(arpd_vals)) if arpd_vals else None
    evaluation["std"] = {
        "success_rate": _std("success_rate"),
        "average_reward": _std("average_reward"),
        "average_steps": _std("average_steps"),
        "arpd": float(np.std(arpd_vals)) if arpd_vals else None,
    }
    evaluation["seed_samples"] = [
        {
            "seed": seed + 1000 + idx,
            "success_rate": s["success_rate"],
            "average_steps": s["average_steps"],
            "arpd": s["arpd"],
        }
        for idx, s in enumerate(seed_samples)
    ]
    evaluation["stochastic_eval"] = apply_stochastic
    evaluation["slip_prob"] = slip
    evaluation["random_start_radius"] = random_radius
    return {
        "maze": env.to_payload(),
        "task_id": task_id,
        "training": training,
        "evaluation": evaluation,
        "q_table": agent.export_q_table(),
    }


def _sample_start_position(env: GridWorldEnv, rng: np.random.Generator, radius: int) -> Tuple[int, int]:
    sx, sy = env.start
    candidates: List[Tuple[int, int]] = []
    for x in range(max(0, sx - radius), min(env.width - 1, sx + radius) + 1):
        for y in range(max(0, sy - radius), min(env.height - 1, sy + radius) + 1):
            pos = (x, y)
            if pos in env.walls or pos in env.lava or pos == env.goal:
                continue
            candidates.append(pos)
    if not candidates:
        return env.start
    return candidates[int(rng.integers(0, len(candidates)))]


def append_history(history_file: Path, entry: Dict) -> None:
    history_file.parent.mkdir(parents=True, exist_ok=True)
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(entry)
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
