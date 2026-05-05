from pathlib import Path

from rl_core.q_learning_agent import run_training_session


def test_training_session_returns_core_metrics():
    reward_path = Path(__file__).resolve().parents[1] / "generated_rewards" / "current_reward.py"
    result = run_training_session(reward_path=reward_path, episodes=30, eval_trials=5)
    assert "training" in result
    assert "evaluation" in result
    assert 0.0 <= result["evaluation"]["success_rate"] <= 1.0
    assert result["evaluation"]["trajectory"]
