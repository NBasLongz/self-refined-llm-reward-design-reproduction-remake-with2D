from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


Position = Tuple[int, int]

DEFAULT_WALLS = {
    (1, 1), (2, 1), (3, 1), (5, 1), (5, 2),
    (5, 3), (1, 5), (2, 5), (3, 5), (4, 5),
}
DEFAULT_LAVA = {(3, 3), (4, 3), (4, 4), (6, 5)}

MAZE_TASKS: Dict[str, Dict] = {
    "open_field": {
        "id": "open_field",
        "name": "Open Field",
        "description": "Easy navigation without hazards; validates that the loop does not over-refine solved tasks.",
        "width": 6,
        "height": 6,
        "start": (0, 0),
        "goal": (5, 5),
        "walls": set(),
        "lava": set(),
        "max_steps": 50,
    },
    "lava_maze": {
        "id": "lava_maze",
        "name": "Lava Maze",
        "description": "Main self-refinement task with lava hazards and walls.",
        "width": 8,
        "height": 8,
        "start": (0, 0),
        "goal": (7, 7),
        "walls": DEFAULT_WALLS,
        "lava": DEFAULT_LAVA,
        "max_steps": 80,
    },
    "wall_corridor": {
        "id": "wall_corridor",
        "name": "Wall Corridor",
        "description": "Narrow-corridor task for wall penalties and path efficiency.",
        "width": 9,
        "height": 9,
        "start": (0, 0),
        "goal": (8, 8),
        "walls": {
            (2, 0), (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 7), (4, 8),
            (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 8),
        },
        "lava": {(1, 7), (5, 2), (7, 4)},
        "max_steps": 95,
    },
    "lava_lake": {
        "id": "lava_lake",
        "name": "Lava Lake",
        "description": "Wide map with a central hazard lake; tests whether reward avoids attractive short unsafe paths.",
        "width": 10,
        "height": 8,
        "start": (0, 3),
        "goal": (9, 3),
        "walls": {(7, 1), (7, 6)},
        "lava": {(4, 2), (4, 3), (4, 4), (4, 5), (5, 2), (5, 3), (5, 4), (5, 5)},
        "max_steps": 100,
    },
    "spiral_maze": {
        "id": "spiral_maze",
        "name": "Zigzag Maze",
        "description": "Longer route with alternating wall gaps and misleading Manhattan-distance shortcuts.",
        "width": 10,
        "height": 10,
        "start": (0, 0),
        "goal": (9, 9),
        "walls": {
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 9),
            (4, 0), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 9),
            (8, 0), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9),
        },
        "lava": {(0, 9), (1, 0), (3, 9), (9, 0)},
        "max_steps": 130,
    },
    "trap_room": {
        "id": "trap_room",
        "name": "Trap Room",
        "description": "Goal is near hazards; tests terminal constraints and local hazard shaping.",
        "width": 8,
        "height": 8,
        "start": (0, 7),
        "goal": (7, 0),
        "walls": {(1, 5), (2, 5), (3, 5), (5, 2), (5, 3), (5, 4), (2, 2), (3, 2)},
        "lava": {(6, 1), (5, 1), (4, 4), (3, 4), (1, 6)},
        "max_steps": 90,
    },
}


@dataclass(frozen=True)
class StepResult:
    observation: Dict
    done: bool
    info: Dict


class GridWorldEnv:
    """Small deterministic 2D maze used as the CPU replacement for Isaac Sim."""

    ACTIONS: Dict[int, Position] = {
        0: (0, -1),  # up
        1: (1, 0),   # right
        2: (0, 1),   # down
        3: (-1, 0),  # left
    }
    ACTION_NAMES: Dict[int, str] = {
        0: "UP",
        1: "RIGHT",
        2: "DOWN",
        3: "LEFT",
    }

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        start: Position = (0, 0),
        goal: Position = (7, 7),
        walls: Optional[Iterable[Position]] = None,
        lava: Optional[Iterable[Position]] = None,
        max_steps: int = 80,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.walls = set(DEFAULT_WALLS if walls is None else walls)
        self.lava = set(DEFAULT_LAVA if lava is None else lava)
        self.max_steps = max_steps
        self.agent_pos = self.start
        self.description = ""
        self.name = ""
        self.step_count = 0
        self.done = False
        self.success = False
        self.last_info: Dict = {}

        blocked = self.walls | self.lava
        if self.start in blocked:
            raise ValueError("Start position cannot be wall or lava.")
        if self.goal in blocked:
            raise ValueError("Goal position cannot be wall or lava.")

    def reset(self) -> Dict:
        self.agent_pos = self.start
        self.step_count = 0
        self.done = False
        self.success = False
        self.last_info = {}
        return self.get_observation()

    def clone(self) -> "GridWorldEnv":
        env = GridWorldEnv(
            width=self.width,
            height=self.height,
            start=self.start,
            goal=self.goal,
            walls=set(self.walls),
            lava=set(self.lava),
            max_steps=self.max_steps,
        )
        env.description = self.description
        env.name = self.name
        return env

    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, pos: Position) -> bool:
        return (not self.in_bounds(pos)) or pos in self.walls

    def is_terminal_hazard(self, pos: Position) -> bool:
        return pos in self.lava

    def step(self, action: int) -> StepResult:
        if self.done:
            return StepResult(self.get_observation(), True, dict(self.last_info))

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid actions: {sorted(self.ACTIONS)}")

        old_pos = self.agent_pos
        dx, dy = self.ACTIONS[action]
        candidate = (old_pos[0] + dx, old_pos[1] + dy)
        hit_wall = self.is_blocked(candidate)
        next_pos = old_pos if hit_wall else candidate

        self.agent_pos = next_pos
        self.step_count += 1

        hit_lava = self.is_terminal_hazard(next_pos)
        reached_goal = next_pos == self.goal
        timed_out = self.step_count >= self.max_steps

        self.success = reached_goal
        self.done = reached_goal or hit_lava or timed_out

        info = {
            "old_position": old_pos,
            "new_position": next_pos,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "hit_wall": hit_wall,
            "hit_lava": hit_lava,
            "reached_goal": reached_goal,
            "timed_out": timed_out,
            "done": self.done,
            "success": self.success,
            "step_count": self.step_count,
        }
        self.last_info = info
        return StepResult(self.get_observation(extra=info), self.done, info)

    def get_observation(self, extra: Optional[Dict] = None) -> Dict:
        distance = self.manhattan(self.agent_pos, self.goal)
        path_distance = self.shortest_path_length_from(self.agent_pos)
        nearest_lava = min((self.manhattan(self.agent_pos, p) for p in self.lava), default=None)
        obs = {
            "position": self.agent_pos,
            "x": self.agent_pos[0],
            "y": self.agent_pos[1],
            "goal": self.goal,
            "goal_x": self.goal[0],
            "goal_y": self.goal[1],
            "lava": sorted(self.lava),
            "walls": sorted(self.walls),
            "width": self.width,
            "height": self.height,
            "distance_to_goal": distance,
            "shortest_path_distance": path_distance if path_distance is not None else distance,
            "nearest_lava_distance": nearest_lava,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
            "success": self.success,
        }
        if extra:
            obs.update(extra)
        return obs

    def to_payload(self) -> Dict:
        return {
            "width": self.width,
            "height": self.height,
            "start": self.start,
            "goal": self.goal,
            "walls": sorted(self.walls),
            "lava": sorted(self.lava),
            "max_steps": self.max_steps,
            "optimal_path_length": self.shortest_path_length(),
            "name": self.name,
            "description": self.description,
        }

    @staticmethod
    def manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def shortest_path_length(self, avoid_lava: bool = True) -> Optional[int]:
        return self.shortest_path_length_from(self.start, avoid_lava=avoid_lava)

    def shortest_path_length_from(self, source: Position, avoid_lava: bool = True) -> Optional[int]:
        blocked = set(self.walls)
        if avoid_lava:
            blocked |= set(self.lava)

        if source in blocked or not self.in_bounds(source):
            return None

        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            pos, dist = queue.popleft()
            if pos == self.goal:
                return dist
            for dx, dy in self.ACTIONS.values():
                nxt = (pos[0] + dx, pos[1] + dy)
                if not self.in_bounds(nxt) or nxt in blocked or nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def render_ascii(self) -> str:
        rows: List[str] = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.agent_pos:
                    row.append("A")
                elif pos == self.goal:
                    row.append("G")
                elif pos == self.start:
                    row.append("S")
                elif pos in self.walls:
                    row.append("#")
                elif pos in self.lava:
                    row.append("L")
                else:
                    row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)


def list_task_payloads() -> List[Dict]:
    payloads = []
    for task_id in MAZE_TASKS:
        env = make_env(task_id)
        spec = MAZE_TASKS[task_id]
        payloads.append(
            {
                "id": task_id,
                "name": spec["name"],
                "description": spec["description"],
                "optimal_path_length": env.shortest_path_length(),
            }
        )
    return payloads


def make_env(task_id: str = "lava_maze") -> GridWorldEnv:
    spec = MAZE_TASKS.get(task_id, MAZE_TASKS["lava_maze"])
    env = GridWorldEnv(
        width=spec["width"],
        height=spec["height"],
        start=spec["start"],
        goal=spec["goal"],
        walls=set(spec["walls"]),
        lava=set(spec["lava"]),
        max_steps=spec["max_steps"],
    )
    env.name = spec.get("name", "")
    env.description = spec.get("description", "")
    return env
