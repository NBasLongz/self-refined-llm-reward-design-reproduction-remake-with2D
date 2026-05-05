def calculate_reward(state, action, next_state, info=None):
    """Human-designed baseline reward for the 2D maze task."""
    info = info or {}
    if info.get("reached_goal") or next_state.get("success"):
        return 120.0
    if info.get("hit_lava"):
        return -140.0
    if info.get("hit_wall"):
        return -12.0
    if info.get("timed_out"):
        return -35.0

    old_distance = state.get("shortest_path_distance", state.get("distance_to_goal", 0))
    new_distance = next_state.get("shortest_path_distance", next_state.get("distance_to_goal", old_distance))
    progress = old_distance - new_distance
    nearest_lava = next_state.get("nearest_lava_distance")

    reward = -1.0 + 6.0 * progress
    if nearest_lava is not None:
        if nearest_lava == 1:
            reward -= 18.0
        elif nearest_lava == 2:
            reward -= 4.0
    return float(reward)
