import functools

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures, mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import jumping_ball

from dmc_memory_maze.maze import MemoryMaze
from dmc_memory_maze.wrappers import (DiscreteActionSetWrapper,
                                      RemapObservationWrapper,
                                      TargetColorAsBorderWrapper)


def test_maze(discrete_actions=True,
              random_state=None, 
              target_color_in_image=True,
              top_camera=False, 
              good_visibility=False,
              ):

    walker = jumping_ball.RollingBallWithHead(
        camera_height=0,
        add_ears=top_camera
    )

    # Build a maze with rooms and targets.
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    arena = mazes.RandomMazeWithTargets(
        x_cells=11,
        y_cells=11,
        xy_scale=2.0,
        z_height=1.2 if not good_visibility else 0.4,
        max_rooms=4,
        room_min_size=4,
        room_max_size=5,
        spawns_per_room=1,
        targets_per_room=1,
        wall_textures=wall_textures,
        skybox_texture=None,  # TODO: remove clouds
        aesthetic='outdoor_natural')

    # Custom memory maze task
    task = MemoryMaze(
        walker=walker,
        maze_arena=arena,
        n_targets=3,
        target_radius=0.3 if not good_visibility else 0.5,
        enable_global_task_observables=True)

    # Built-in task
    # task = random_goal_maze.ManyGoalsMaze(
    #     walker=walker,
    #     maze_arena=arena,
    #     target_builder=functools.partial(
    #         target_sphere.TargetSphere,
    #         radius=0.3,
    #         height_above_ground=.3,
    #         rgb1=(0, 0, 0.4),
    #         rgb2=(0, 0, 0.7)),
    #     target_reward_scale=1.,
    #     contact_termination=False,
    #     physics_timestep=0.005,
    #     control_timestep=0.050)

    if top_camera:
        task.observables['top_camera'].enabled = True

    env = composer.Environment(
        time_limit=30,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

    camera_key = 'walker/egocentric_camera' if not top_camera else 'top_camera'
    env = RemapObservationWrapper(env, {
        'image': camera_key,
        'target_color': 'target_color',
    })

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0., 0.]),  # noop
            np.array([-1., 0.]),  # forward
            np.array([0., -1.]),  # left
            np.array([0., +1.]),  # right
            np.array([-1., -1.]),  # forward + left
            np.array([-1., +1.]),  # forward + right
        ])

    return env
