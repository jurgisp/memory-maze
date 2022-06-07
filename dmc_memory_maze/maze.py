import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze

DEFAULT_CONTROL_TIMESTEP = 0.050    # From jumping_ball_test  # DEFAULT_CONTROL_TIMESTEP = 0.025
DEFAULT_PHYSICS_TIMESTEP = 0.005    # From jumping_ball_test  # DEFAULT_PHYSICS_TIMESTEP = 0.001


class MemoryMaze(random_goal_maze.NullGoalMaze):
    # Adapted from dm_control.locomotion.tasks.RepeatSingleGoalMaze

    def __init__(self,
                 walker,
                 maze_arena,
                 target=None,
                 target_reward_scale=1.0,
                 randomize_spawn_position=True,
                 randomize_spawn_rotation=True,
                 rotation_bias_factor=0,
                 aliveness_reward=0.0,
                 aliveness_threshold=-0.5,
                 contact_termination=True,
                 enable_global_task_observables=False,
                 physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep=DEFAULT_CONTROL_TIMESTEP,
                 ):
        super().__init__(
            walker=walker,
            maze_arena=maze_arena,
            randomize_spawn_position=randomize_spawn_position,
            randomize_spawn_rotation=randomize_spawn_rotation,
            rotation_bias_factor=rotation_bias_factor,
            aliveness_reward=aliveness_reward,
            aliveness_threshold=aliveness_threshold,
            contact_termination=contact_termination,
            enable_global_task_observables=enable_global_task_observables,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep)
        if target is None:
            target = target_sphere.TargetSphere()
        self._target = target
        self._rewarded_this_step = False
        self._maze_arena.attach(target)
        self._target_reward_scale = target_reward_scale
        self._targets_obtained = 0

        if enable_global_task_observables:
            xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

            def _target_pos(physics, target=target):
                return physics.bind(target.geom).xpos

            walker.observables.add_egocentric_vector(
                'target_0',
                observable_lib.Generic(_target_pos),
                origin_callable=xpos_origin_callable)

    def initialize_episode_mjcf(self, random_state):
        super().initialize_episode_mjcf(random_state)
        ix = random_state.randint(0, len(self._maze_arena.target_positions))
        self._target_position = self._maze_arena.target_positions[ix]
        mjcf.get_attachment_frame(self._target.mjcf_model).pos = self._target_position

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self._rewarded_this_step = False
        self._targets_obtained = 0

    def after_step(self, physics, random_state):
        super().after_step(physics, random_state)
        if self._target.activated:
            self._rewarded_this_step = True
            self._targets_obtained += 1
            self._respawn(physics, random_state)
            self._target.reset(physics)
        else:
            self._rewarded_this_step = False

    def should_terminate_episode(self, physics):
        if super().should_terminate_episode(physics):
            return True

    def get_reward(self, physics):
        del physics
        if self._rewarded_this_step:
            target_reward = self._target_reward_scale
        else:
            target_reward = 0.0
        return target_reward + self._aliveness_reward
