from collections import deque
from typing import List, Optional, Tuple
import numpy as np

from memory_maze.wrappers import ObservationWrapper


class PathToTargetWrapper(ObservationWrapper):
    """Find shortest path to target and indicate it on maze_layout. Used for Oracle."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert 'agent_pos' in spec
        assert 'target_pos' in spec
        assert 'maze_layout' in spec
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        # Find shortest path (in gridworld) from agent to target
        maze = obs['maze_layout']
        start = tuple(obs['agent_pos'].astype(int))
        finish = tuple(obs['target_pos'].astype(int))
        path = breadth_first_search(maze, start, finish)
        if path:
            for x, y in path:
                maze[y, x] = 2  # Update maze_layout observation
        return obs


class DrawMinimapWrapper(ObservationWrapper):
    """Show maze_layout as minimap in image observation. Used for Oracle."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert 'maze_layout' in spec
        assert 'image' in spec
        assert 'agent_dir' in spec
        return spec

    def observation(self, obs):
        from PIL import Image

        assert isinstance(obs, dict)
        maze = obs['maze_layout']
        x, y = obs['agent_pos']
        dx, dy = obs['agent_dir']
        angle = np.arctan2(dx, dy)
        N = maze.shape[0]
        SIZE = N * 2

        # Draw map
        map = np.zeros((N, N, 3), np.uint8)  # walls in black
        map[:, :] += (maze == 1)[..., None] * np.array([[[255, 255, 255]]], np.uint8)  # corridors in white
        map[:, :] += (maze == 2)[..., None] * np.array([[[0, 255, 0]]], np.uint8)  # path in green
        map[int(y), int(x)] = np.array([255, 0, 0], np.uint8)  # agent in red
        map = np.flip(map, 0)

        # Scale, rotate, translate
        mapimg = Image.fromarray(map)
        mapimg = mapimg.resize((SIZE, SIZE), resample=0)
        tx = (x - N / 2) / N * SIZE
        ty = - (y - N / 2) / N * SIZE
        mapimg = mapimg.transform(mapimg.size, 0,
                                  (1, 0, tx,
                                   0, 1, ty),
                                  resample=0)
        mapimg = mapimg.rotate(angle / np.pi * 180, resample=0)

        # Overlay minimap onto observation image top-right corner
        img = obs['image']
        img[:SIZE, -SIZE:] = img[:SIZE, -SIZE:] // 2 + np.array(mapimg) // 2
        return obs


def breadth_first_search(maze: np.ndarray, start: Tuple[int, int], finish: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    h, w = maze.shape

    queue = deque()
    visited = np.zeros(maze.shape, dtype=bool)
    backtrace = np.zeros(maze.shape + (2,), dtype=int)

    xs, ys = start
    queue.append((xs, ys))
    visited[ys, xs] = True

    while len(queue) > 0:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x1 = x + dx
            y1 = y + dy
            if 0 <= x1 < w and 0 <= y1 < h and maze[y1, x1] and not visited[y1, x1]:
                queue.append((x1, y1))
                visited[y1, x1] = True
                backtrace[y1, x1, :] = np.array([x, y])
                if (x1, y1) == finish:
                    break

    xf, yf = finish
    if not visited[yf, xf]:
        return None

    path = []
    path.append((xf, yf))
    while (xf, yf) != start:
        xf, yf = backtrace[yf, xf]
        path.append((xf, yf))
    path.reverse()
    return path
