import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from numpy.random import RandomState

DEFAULT_CONTROL_TIMESTEP = 0.025
DEFAULT_PHYSICS_TIMESTEP = 0.005

TARGET_COLORS = [
    np.array([1.0, 0.0, 0.0]),  # red
    np.array([0.0, 1.0, 0.0]),  # green
    np.array([0.0, 0.0, 1.0]),  # blue
    np.array([0.44, 0.15, 0.76]),  # purple
    np.array([1.00, 1.00, 0.00]),  # yellow
    np.array([0.00, 1.00, 1.00]),  # cyan
]


class MemoryMaze(random_goal_maze.NullGoalMaze):
    # Adapted from dm_control.locomotion.tasks.RepeatSingleGoalMaze

    def __init__(self,
                 walker,
                 maze_arena,
                 n_targets=3,
                 target_radius=0.3,
                 target_reward_scale=1.0,
                 enable_global_task_observables=False,
                 physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
                 control_timestep=DEFAULT_CONTROL_TIMESTEP,
                 ):
        super().__init__(
            walker=walker,
            maze_arena=maze_arena,
            randomize_spawn_position=True,
            randomize_spawn_rotation=True,
            contact_termination=False,
            enable_global_task_observables=enable_global_task_observables,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep
        )
        self._target_reward_scale = target_reward_scale
        self._targets = []
        for i in range(n_targets):
            color = TARGET_COLORS[i]
            target = target_sphere.TargetSphere(
                radius=target_radius,
                height_above_ground=target_radius,
                rgb1=tuple(color * 0.7),
                rgb2=tuple(color * 0.4),
            )
            self._targets.append(target)
            self._maze_arena.attach(target)
        self._current_target_ix = 0
        self._rewarded_this_step = False
        self._targets_obtained = 0

        # if enable_global_task_observables:  # TODO: probe vectors
        #     xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

        #     def _target_pos(physics, target=target):
        #         return physics.bind(target.geom).xpos

        #     walker.observables.add_egocentric_vector(
        #         'target_0',
        #         observable_lib.Generic(_target_pos),
        #         origin_callable=xpos_origin_callable)

        self._task_observables = super().task_observables
        target_color_obs = observable_lib.Generic(
            lambda _: TARGET_COLORS[self._current_target_ix])
        target_color_obs.enabled = True
        self._task_observables['target_color'] = target_color_obs

    @property
    def task_observables(self):
        return self._task_observables

    @property
    def name(self):
        return 'memory_maze'

    def initialize_episode_mjcf(self, rng: RandomState):
        super().initialize_episode_mjcf(rng)
        while True:
            ok = self._place_targets(rng)
            if not ok:
                # Could not place targets - regenerate the maze
                self._maze_arena.regenerate()
                continue
            break
        self._pick_new_target(rng)

    def initialize_episode(self, physics, rng: RandomState):
        super().initialize_episode(physics, rng)
        self._rewarded_this_step = False
        self._targets_obtained = 0

    def after_step(self, physics, rng: RandomState):
        super().after_step(physics, rng)
        self._rewarded_this_step = False
        for i, target in enumerate(self._targets):
            if target.activated:
                if i == self._current_target_ix:
                    self._rewarded_this_step = True
                    self._targets_obtained += 1
                    self._pick_new_target(rng)
                target.reset(physics)  # Resets activated=False

    def should_terminate_episode(self, physics):
        return super().should_terminate_episode(physics)

    def get_reward(self, physics):
        if self._rewarded_this_step:
            return self._target_reward_scale
        return 0.0

    def _place_targets(self, rng: RandomState) -> bool:
        possible_positions = list(self._maze_arena.target_positions)
        rng.shuffle(possible_positions)
        if len(possible_positions) < len(self._targets):
            # Too few rooms - need to regenerate the maze
            return False
        for target, pos in zip(self._targets, possible_positions):
            mjcf.get_attachment_frame(target.mjcf_model).pos = pos
        return True

    def _pick_new_target(self, rng: RandomState):
        while True:
            ix = rng.randint(len(self._targets))
            if self._targets[ix].activated:
                continue  # Skip the target that the agent is touching
            self._current_target_ix = ix
            break
