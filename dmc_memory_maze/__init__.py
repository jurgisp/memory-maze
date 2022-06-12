
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

    sizes = {
        '9x9': tasks.memory_maze_9x9,
        '11x11': tasks.memory_maze_11x11,
        '13x13': tasks.memory_maze_13x13,
        '15x15': tasks.memory_maze_15x15,
    }

    for key, dm_task in sizes.items():
        register(id=f'MemoryMaze-{key}-v0', entry_point=f(_make_gym_env, dm_task))
        register(id=f'MemoryMaze-{key}-Vis-v0', entry_point=f(_make_gym_env, dm_task, good_visibility=True))
        register(id=f'MemoryMaze-{key}-Top-v0', entry_point=f(_make_gym_env, dm_task, top_camera=True))
        register(id=f'MemoryMaze-{key}-HiFps-v0', entry_point=f(_make_gym_env, dm_task, control_fps=40))
        register(id=f'MemoryMaze-{key}-HiFps-Vis-v0', entry_point=f(_make_gym_env, dm_task, good_visibility=True, control_fps=40))
        register(id=f'MemoryMaze-{key}-HiFps-Top-v0', entry_point=f(_make_gym_env, dm_task, top_camera=True, control_fps=40))


except ImportError:
    print('dmc_memory_maze: gym environments not registered.')
    raise
