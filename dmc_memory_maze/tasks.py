import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures, mazes

from dmc_memory_maze.maze import MemoryMaze, RollingBallWithFriction
from dmc_memory_maze.wrappers import (DiscreteActionSetWrapper,
                                      RemapObservationWrapper,
                                      TargetColorAsBorderWrapper)

# Slow control (4Hz), so that agent without HRL has a chance.
# Native control would be 40Hz, so this corresponds roughly to action_repeat=10.
DEFAULT_CONTROL_FREQ = 4

def memory_maze_9x9(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_small
    {
        mazeHeight = 11,  # with outer walls
        mazeWidth = 11,
        roomCount = 4,
        roomMaxSize = 5,
        roomMinSize = 3,
    }
    """
    return _memory_maze(9, 3, 250, **kwargs)

def memory_maze_11x11(**kwargs):
    return _memory_maze(11, 4, 500, **kwargs)

def memory_maze_13x13(**kwargs):
    return _memory_maze(13, 5, 750, **kwargs)

def memory_maze_15x15(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_large
    {
        mazeHeight = 17,  # with outer walls
        mazeWidth = 17,
        roomCount = 9,
        roomMaxSize = 3,
        roomMaxSize = 3,
    }
    """
    return _memory_maze(15, 6, 1000, max_rooms=9, room_max_size=3, **kwargs)

def _memory_maze(
    maze_size,  # measured without exterior walls
    n_targets,
    time_limit,
    max_rooms=6,
    room_min_size=3,
    room_max_size=5,
    control_freq=DEFAULT_CONTROL_FREQ,
    discrete_actions=True,
    target_color_in_image=True,
    top_camera=False,
    good_visibility=False,
    random_state=None,
):
    walker = RollingBallWithFriction(camera_height=0, add_ears=top_camera)

    wall_textures = labmaze_textures.WallTextures(style='style_01')
    arena = mazes.RandomMazeWithTargets(
        x_cells=maze_size + 2,
        y_cells=maze_size + 2,
        xy_scale=2.0,
        z_height=1.2 if not good_visibility else 0.4,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        spawns_per_room=1,
        targets_per_room=1,
        wall_textures=wall_textures,
        skybox_texture=None,  # TODO: remove clouds
        aesthetic='outdoor_natural',
    )

    # Custom memory maze task
    task = MemoryMaze(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.25 if not good_visibility else 0.5,
        enable_global_task_observables=True,
        control_timestep=1.0 / control_freq
    )

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
        time_limit=time_limit - 1e-3,  # subtract epsilon to make sure ep_length=time_limit*fps
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)

    env = RemapObservationWrapper(env, {
        'image': 'walker/egocentric_camera' if not top_camera else 'top_camera',
        'target_color': 'target_color',
    })

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(env, [
            np.array([0.0, 0.0]),  # noop
            np.array([-1.0, 0.0]),  # forward
            np.array([0.0, -1.0]),  # left
            np.array([0.0, +1.0]),  # right
            np.array([-1.0, -1.0]),  # forward + left
            np.array([-1.0, +1.0]),  # forward + right
        ])

    return env
