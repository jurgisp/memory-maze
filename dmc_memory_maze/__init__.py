
from . import tasks

try:
    # Register gym environments, if gym is available

    from typing import Callable
    import functools

    import dm_env
    import gym.envs.registration

    from .gym_wrappers import GymWrapper

    def _make_gym_env(ctor: Callable[[], dm_env.Environment]):
        dmenv = ctor()
        return GymWrapper(dmenv)

    gym.envs.registration.register(
        id="MemMaze-9x9-v0",
        entry_point=functools.partial(_make_gym_env, ctor=tasks.test_maze)
    )


except ImportError:
    print('dmc_memory_maze: gym environments not registered.')
    raise
