import functools

import numpy as np
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import (bowl, corridors, floors, labmaze_textures, mazes)
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import jumping_ball

from dmc_memory_maze.wrappers import (DiscreteActionSetWrapper,
                                      RemapObservationWrapper)

_CONTROL_TIMESTEP = 0.020    # From jumping_ball_test  # DEFAULT_CONTROL_TIMESTEP = 0.025
_PHYSICS_TIMESTEP = 0.005    # From jumping_ball_test  # DEFAULT_PHYSICS_TIMESTEP = 0.001


def test_maze(discrete_actions=True, random_state=None):

    walker = jumping_ball.RollingBallWithHead(
        observable_options={'egocentric_camera': dict(enabled=True)}
    )

    # Build a maze with rooms and targets.
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    arena = mazes.RandomMazeWithTargets(
        x_cells=11,
        y_cells=11,
        xy_scale=.5,
        z_height=.3,
        max_rooms=4,
        room_min_size=4,
        room_max_size=5,
        spawns_per_room=1,
        targets_per_room=3,
        wall_textures=wall_textures,
        aesthetic='outdoor_natural')

    # Build a task that rewards the agent for obtaining targets.
    task = random_goal_maze.ManyGoalsMaze(
        walker=walker,
        maze_arena=arena,
        target_builder=functools.partial(
            target_sphere.TargetSphere,
            radius=0.05,
            height_above_ground=.125,
            rgb1=(0, 0, 0.4),
            rgb2=(0, 0, 0.7)),
        target_reward_scale=50.,
        contact_termination=False,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    env = composer.Environment(time_limit=30,
                               task=task,
                               random_state=random_state,
                               strip_singleton_obs_buffer_dim=True)

    env = RemapObservationWrapper(env, {'image': 'walker/egocentric_camera'})

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0., 0.]),  # noop
            np.array([-1., 0.]),  # move forward
            np.array([0., -1.]),  # turn left
            np.array([0., +1.]),  # turn right
            np.array([+1., 0.]),  # move backward
        ])

    return env
