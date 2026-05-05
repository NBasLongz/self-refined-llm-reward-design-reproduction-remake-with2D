def calculate_reward(state, action, next_state, info=None):
    """
    Reward function for the Lava Lake environment.

    Parameters
    ----------
    state : dict
        Current observation. Must contain a ``pos`` key with a (x, y) tuple.
    action : any
        Action taken by the agent (unused in this reward).
    next_state : dict
        Observation after the action. Must contain ``pos`` and may contain
        ``reached_goal``, ``hit_lava`` and ``hit_wall`` booleans.
    info : dict, optional
        Additional information from the environment step (ignored).

    Returns
    -------
    float
        Reward for the transition.
    """
    # Environment constants
    GOAL_POS = (9, 3)          # hard‑coded goal for this map
    STEP_PENALTY = -0.05       # mild cost per step
    GOAL_REWARD = 20.0         # large reward for reaching the goal
    LAVA_PENALTY = -15.0       # strong penalty for stepping into lava
    WALL_PENALTY = -5.0        # penalty for hitting a wall
    DIST_COEFF = 0.8           # reward for getting closer to the goal

    # Helper: Manhattan distance to goal
    def manhattan(pos):
        return abs(pos[0] - GOAL_POS[0]) + abs(pos[1] - GOAL_POS[1])

    # Current and next distances
    curr_pos = state.get("pos", (0, 0))
    next_pos = next_state.get("pos", (0, 0))
    curr_dist = manhattan(curr_pos)
    next_dist = manhattan(next_pos)

    # Distance improvement reward (positive if closer, negative if farther)
    dist_reward = DIST_COEFF * (curr_dist - next_dist)

    # Base reward: step penalty + distance reward
    reward = STEP_PENALTY + dist_reward

    # Goal reached
    if next_state.get("reached_goal", False):
        reward += GOAL_REWARD

    # Lava hit
    if next_state.get("hit_lava", False):
        reward += LAVA_PENALTY

    # Wall hit
    if next_state.get("hit_wall", False):
        reward += WALL_PENALTY

    return float(reward)
