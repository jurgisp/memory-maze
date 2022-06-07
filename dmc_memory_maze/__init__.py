
from . import tasks

try:
    # Register gym environments, if gym is available

    from typing import Callable
    from functools import partial as f

    import dm_env
    import gym.envs.registration

    from .gym_wrappers import GymWrapper

    def _make_gym_env(dm_task: Callable[[], dm_env.Environment], **kwargs):
        dmenv = dm_task(**kwargs)
        return GymWrapper(dmenv)

    gym.envs.registration.register(id="MemMaze-9x9-v0", entry_point=f(_make_gym_env, tasks.test_maze))
    gym.envs.registration.register(id="MemMaze-9x9-Top-v0", entry_point=f(_make_gym_env, tasks.test_maze, top_camera=True))
    gym.envs.registration.register(id="MemMaze-9x9-Low-v0", entry_point=f(_make_gym_env, tasks.test_maze, low_walls=True))


except ImportError:
    print('dmc_memory_maze: gym environments not registered.')
    raise
