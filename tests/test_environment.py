from rl_core.environment import GridWorldEnv


def test_reset_returns_start_position():
    env = GridWorldEnv()
    obs = env.reset()
    assert obs["position"] == env.start
    assert obs["distance_to_goal"] > 0


def test_wall_keeps_agent_in_place():
    env = GridWorldEnv(width=3, height=3, start=(0, 0), goal=(2, 2), walls={(1, 0)}, lava=set())
    env.reset()
    result = env.step(1)
    assert result.info["hit_wall"] is True
    assert result.observation["position"] == (0, 0)
    


def test_lava_terminates_episode():
    env = GridWorldEnv(width=3, height=3, start=(0, 0), goal=(2, 2), walls=set(), lava={(1, 0)})
    env.reset()
    result = env.step(1)
    assert result.done is True
    assert result.info["hit_lava"] is True


def test_shortest_path_exists_for_default_maze():
    env = GridWorldEnv()
    assert env.shortest_path_length() is not None
