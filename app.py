from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request

from config import (
    ARPD_THRESHOLD,
    DEFAULT_EPISODES,
    DEFAULT_EVAL_TRIALS,
    EVAL_MULTI_SEEDS,
    EVAL_RANDOM_START_RADIUS,
    EVAL_SLIP_PROB,
    STOCHASTIC_EVAL,
    HISTORY_FILE,
    INITIAL_EPISODES_FACTOR,
    MANUAL_REWARD_FILE,
    MAX_REFINEMENT_ITERATIONS,
    REWARD_FILE,
    SUCCESS_THRESHOLD,
)
from llm_module import LLMRewardDesigner
from rl_core.environment import list_task_payloads, make_env
from rl_core.q_learning_agent import append_history, run_training_session
try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


app = Flask(__name__)
designer = LLMRewardDesigner()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_run(label: str, generation: Dict, run: Dict, iteration: int) -> Dict:
    evaluation = run["evaluation"]
    training = run["training"]
    return {
        "timestamp": now_iso(),
        "label": label,
        "iteration": iteration,
        "reward_source": generation.get("source"),
        "llm_latency_sec": generation.get("latency_sec", 0.0),
        "success_rate": evaluation["success_rate"],
        "average_reward": evaluation["average_reward"],
        "average_steps": evaluation["average_steps"],
        "lava_hits": evaluation["lava_hits"],
        "wall_hits": evaluation["wall_hits"],
        "arpd": evaluation["arpd"],
        "convergence_episode": training["convergence_episode"],
        "cpu_training_time_sec": training["cpu_training_time_sec"],
        "cpu_eval_time_sec": evaluation["cpu_eval_time_sec"],
    }


def run_self_refine_for_task(
    task_id: str,
    episodes: int,
    eval_trials: int,
    max_iterations: int,
    threshold: float,
    arpd_threshold: float,
) -> Dict:
    env = make_env(task_id)
    history = []
    initial_episodes = max(80, int(episodes * INITIAL_EPISODES_FACTOR))

    generation = designer.generate_initial_reward(env.to_payload())
    designer.save_reward_code(generation["code"])
    run = run_training_session(
        REWARD_FILE,
        episodes=initial_episodes,
        eval_trials=eval_trials,
        seed=42,
        task_id=task_id,
        eval_multi_seeds=EVAL_MULTI_SEEDS,
        stochastic_eval=STOCHASTIC_EVAL,
        eval_slip_prob=EVAL_SLIP_PROB,
        eval_random_start_radius=EVAL_RANDOM_START_RADIUS,
    )
    summary = summarize_run("R_initial", generation, run, iteration=0)
    history.append({**summary, "code": generation["code"], "run": run})

    def needs_refine(eval_data: Dict) -> bool:
        success_ok = eval_data["success_rate"] >= threshold
        arpd = eval_data.get("arpd")
        arpd_ok = arpd is not None and arpd <= arpd_threshold
        return not (success_ok and arpd_ok)

    iteration = 0
    while needs_refine(run["evaluation"]) and iteration < max_iterations:
        iteration += 1
        metrics = {
            "success_rate": run["evaluation"]["success_rate"],
            "average_steps": run["evaluation"]["average_steps"],
            "average_reward": run["evaluation"]["average_reward"],
            "lava_hits": run["evaluation"]["lava_hits"],
            "wall_hits": run["evaluation"]["wall_hits"],
            "arpd": run["evaluation"]["arpd"],
            "optimal_path_length": run["evaluation"].get("optimal_path_length"),
            "target_arpd_threshold": arpd_threshold,
        }
        previous_code = designer.read_current_reward()
        generation = designer.refine_reward(env.to_payload(), metrics, previous_code)
        designer.save_reward_code(generation["code"])
        run = run_training_session(
            REWARD_FILE,
            episodes=episodes,
            eval_trials=eval_trials,
            seed=42 + iteration,
            task_id=task_id,
            eval_multi_seeds=EVAL_MULTI_SEEDS,
            stochastic_eval=STOCHASTIC_EVAL,
            eval_slip_prob=EVAL_SLIP_PROB,
            eval_random_start_radius=EVAL_RANDOM_START_RADIUS,
        )
        summary = summarize_run("R_refined", generation, run, iteration=iteration)
        history.append({**summary, "code": generation["code"], "run": run})

    manual_run = run_training_session(
        MANUAL_REWARD_FILE,
        episodes=episodes,
        eval_trials=eval_trials,
        seed=2023,
        task_id=task_id,
        eval_multi_seeds=EVAL_MULTI_SEEDS,
        stochastic_eval=STOCHASTIC_EVAL,
        eval_slip_prob=EVAL_SLIP_PROB,
        eval_random_start_radius=EVAL_RANDOM_START_RADIUS,
    )
    final = history[-1]
    table_row = {
        "task": task_id,
        "R_initial": history[0]["success_rate"],
        "R_refined": final["success_rate"],
        "R_manual": manual_run["evaluation"]["success_rate"],
        "iterations": max(0, len(history) - 1),
        "avg_steps": final["average_steps"],
        "arpd": final["arpd"],
        "cpu_training_time_sec": final["cpu_training_time_sec"],
        "std_success_rate": final["run"]["evaluation"].get("std", {}).get("success_rate"),
        "std_arpd": final["run"]["evaluation"].get("std", {}).get("arpd"),
    }
    init_samples = history[0]["run"]["evaluation"].get("seed_samples", [])
    refined_samples = final["run"]["evaluation"].get("seed_samples", [])
    p_value = None
    if wilcoxon and init_samples and refined_samples and len(init_samples) == len(refined_samples):
        init_sr = [float(x["success_rate"]) for x in init_samples]
        ref_sr = [float(x["success_rate"]) for x in refined_samples]
        try:
            p_value = float(wilcoxon(init_sr, ref_sr, zero_method="zsplit").pvalue)
        except Exception:
            p_value = None
    return {
        "threshold": threshold,
        "arpd_threshold": arpd_threshold,
        "history": history,
        "manual": manual_run,
        "table_row": table_row,
        "stats": {
            "wilcoxon_p_value_sr_initial_vs_refined": p_value,
            "eval_multi_seeds": EVAL_MULTI_SEEDS,
        },
        "reward_file": str(REWARD_FILE),
        "history_file": str(HISTORY_FILE),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/report")
def report():
    return render_template("report.html")


def request_task_id(default: str = "lava_maze") -> str:
    if request.method == "GET":
        return request.args.get("task_id", default)
    payload = request.get_json(silent=True) or {}
    return payload.get("task_id", default)


@app.get("/api/tasks")
def get_tasks():
    return jsonify(list_task_payloads())


@app.get("/api/maze")
def get_maze():
    task_id = request_task_id()
    payload = make_env(task_id).to_payload()
    payload["task_id"] = task_id
    return jsonify(payload)


@app.get("/api/reward")
def get_reward():
    return jsonify({"code": designer.read_current_reward(), "path": str(REWARD_FILE)})


@app.get("/api/config")
def get_config():
    return jsonify(
        {
            "default_episodes": DEFAULT_EPISODES,
            "default_eval_trials": DEFAULT_EVAL_TRIALS,
            "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
            "success_threshold": SUCCESS_THRESHOLD,
            "arpd_threshold": ARPD_THRESHOLD,
            "eval_multi_seeds": EVAL_MULTI_SEEDS,
            "stochastic_eval": STOCHASTIC_EVAL,
            "eval_slip_prob": EVAL_SLIP_PROB,
            "eval_random_start_radius": EVAL_RANDOM_START_RADIUS,
        }
    )


@app.post("/api/generate-reward")
def generate_reward():
    payload = request.get_json(silent=True) or {}
    task_id = payload.get("task_id", "lava_maze")
    instruction = payload.get("instruction", "")
    env = make_env(task_id)
    mode = payload.get("mode", "initial")

    try:
        if mode == "refine":
            metrics = payload.get("metrics", {})
            previous_code = payload.get("previous_code") or designer.read_current_reward()
            result = designer.refine_reward(env.to_payload(), metrics, previous_code, instruction)
        else:
            result = designer.generate_initial_reward(env.to_payload(), instruction)
            
        designer.save_reward_code(result["code"])
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "mode": mode, "task_id": task_id}), 500


@app.post("/api/run-agent")
def run_agent():
    payload = request.get_json(silent=True) or {}
    episodes = int(payload.get("episodes", DEFAULT_EPISODES))
    eval_trials = int(payload.get("eval_trials", DEFAULT_EVAL_TRIALS))
    seed = int(payload.get("seed", 42))
    task_id = payload.get("task_id", "lava_maze")

    start = time.perf_counter()
    run = run_training_session(
        REWARD_FILE,
        episodes=episodes,
        eval_trials=eval_trials,
        seed=seed,
        task_id=task_id,
        eval_multi_seeds=EVAL_MULTI_SEEDS,
        stochastic_eval=STOCHASTIC_EVAL,
        eval_slip_prob=EVAL_SLIP_PROB,
        eval_random_start_radius=EVAL_RANDOM_START_RADIUS,
    )
    run["total_wall_time_sec"] = time.perf_counter() - start

    entry = {
        "timestamp": now_iso(),
        "type": "single_run",
        "episodes": episodes,
        "eval_trials": eval_trials,
        "task_id": task_id,
        "evaluation": run["evaluation"],
        "training": {
            key: value
            for key, value in run["training"].items()
            if key not in {"episode_returns", "episode_steps"}
        },
    }
    append_history(HISTORY_FILE, entry)
    return jsonify(run)


@app.post("/api/manual-baseline")
def manual_baseline():
    payload = request.get_json(silent=True) or {}
    episodes = int(payload.get("episodes", DEFAULT_EPISODES))
    eval_trials = int(payload.get("eval_trials", DEFAULT_EVAL_TRIALS))
    seed = int(payload.get("seed", 2023))
    task_id = payload.get("task_id", "lava_maze")

    run = run_training_session(
        MANUAL_REWARD_FILE,
        episodes=episodes,
        eval_trials=eval_trials,
        seed=seed,
        task_id=task_id,
        eval_multi_seeds=EVAL_MULTI_SEEDS,
        stochastic_eval=STOCHASTIC_EVAL,
        eval_slip_prob=EVAL_SLIP_PROB,
        eval_random_start_radius=EVAL_RANDOM_START_RADIUS,
    )
    entry = {
        "timestamp": now_iso(),
        "type": "manual_baseline",
        "episodes": episodes,
        "eval_trials": eval_trials,
        "task_id": task_id,
        "evaluation": run["evaluation"],
        "training": {
            key: value
            for key, value in run["training"].items()
            if key not in {"episode_returns", "episode_steps"}
        },
    }
    append_history(HISTORY_FILE, entry)
    return jsonify(run)


@app.post("/api/self-refine")
def self_refine():
    payload = request.get_json(silent=True) or {}
    episodes = int(payload.get("episodes", DEFAULT_EPISODES))
    eval_trials = int(payload.get("eval_trials", DEFAULT_EVAL_TRIALS))
    max_iterations = int(payload.get("max_iterations", MAX_REFINEMENT_ITERATIONS))
    threshold = float(payload.get("success_threshold", SUCCESS_THRESHOLD))
    arpd_threshold = float(payload.get("arpd_threshold", ARPD_THRESHOLD))
    task_id = payload.get("task_id", "lava_maze")

    result = run_self_refine_for_task(task_id, episodes, eval_trials, max_iterations, threshold, arpd_threshold)
    history = result["history"]
    table_row = result["table_row"]

    append_history(
        HISTORY_FILE,
        {
            "timestamp": now_iso(),
            "type": "self_refine",
            "task_id": task_id,
            "threshold": threshold,
            "arpd_threshold": arpd_threshold,
            "table_row": table_row,
            "summaries": [
                {key: value for key, value in item.items() if key not in {"code", "run"}}
                for item in history
            ],
        },
    )

    return jsonify(result)


@app.post("/api/run-all")
def run_all_tasks():
    payload = request.get_json(silent=True) or {}
    episodes = int(payload.get("episodes", DEFAULT_EPISODES))
    eval_trials = int(payload.get("eval_trials", DEFAULT_EVAL_TRIALS))
    max_iterations = int(payload.get("max_iterations", MAX_REFINEMENT_ITERATIONS))
    threshold = float(payload.get("success_threshold", SUCCESS_THRESHOLD))
    arpd_threshold = float(payload.get("arpd_threshold", ARPD_THRESHOLD))
    task_ids = payload.get("task_ids") or [task["id"] for task in list_task_payloads()]

    started = time.perf_counter()
    results = []
    for task_id in task_ids:
        result = run_self_refine_for_task(task_id, episodes, eval_trials, max_iterations, threshold, arpd_threshold)
        results.append(
            {
                "task_id": task_id,
                "table_row": result["table_row"],
                "final_summary": {
                    key: value
                    for key, value in result["history"][-1].items()
                    if key not in {"code", "run"}
                },
            }
        )

    payload_out = {
        "timestamp": now_iso(),
        "type": "run_all",
        "episodes": episodes,
        "eval_trials": eval_trials,
        "threshold": threshold,
        "arpd_threshold": arpd_threshold,
        "results": results,
        "wall_time_sec": time.perf_counter() - started,
    }
    append_history(HISTORY_FILE, payload_out)
    return jsonify(payload_out)


@app.get("/api/history")
def get_history():
    if not Path(HISTORY_FILE).exists():
        return jsonify([])
    try:
        return jsonify(json.loads(Path(HISTORY_FILE).read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return jsonify([])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
