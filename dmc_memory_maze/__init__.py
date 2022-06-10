
from . import tasks

try:
    # Register gym environments, if gym is available

    from typing import Callable
    from functools import partial as f

    import dm_env
    import gym
    from gym.envs.registration import register

    from .gym_wrappers import GymWrapper

    def _make_gym_env(dm_task: Callable[[], dm_env.Environment], **kwargs):
        dmenv = dm_task(**kwargs)
        return GymWrapper(dmenv)

    register(id="MemoryMaze-9x9-v0", entry_point=f(_make_gym_env, tasks.memory_maze_9x9))
    register(id="MemoryMaze-9x9-Top-v0", entry_point=f(_make_gym_env, tasks.memory_maze_9x9, top_camera=True))
    register(id="MemoryMaze-9x9-Vis-v0", entry_point=f(_make_gym_env, tasks.memory_maze_9x9, good_visibility=True))


except ImportError:
    print('dmc_memory_maze: gym environments not registered.')
    raise
