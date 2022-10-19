import functools
import string

import labmaze
import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.arenas import covering, labmaze_textures, mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import jumping_ball
from labmaze import assets as labmaze_assets
from numpy.random import RandomState

DEFAULT_CONTROL_TIMESTEP = 0.025
DEFAULT_PHYSICS_TIMESTEP = 0.005

TARGET_COLORS = [
    np.array([170, 38, 30]) / 220,  # red
    np.array([99, 170, 88]) / 220,  # green
    np.array([39, 140, 217]) / 220,  # blue
    np.array([93, 105, 199]) / 220,  # purple
    np.array([220, 193, 59]) / 220,  # yellow
    np.array([220, 128, 107]) / 220,  # salmon
]


class RollingBallWithFriction(jumping_ball.RollingBallWithHead):

    def _build(self, roll_damping=5.0, steer_damping=20.0, **kwargs):
        super()._build(**kwargs)
        # Increase friction to the joints, so the movement feels more like traditional
        # first-person navigation control, without much acceleration/deceleration.
        self._mjcf_root.find('joint', 'roll').damping = roll_damping
        self._mjcf_root.find('joint', 'steer').damping = steer_damping


class MemoryMazeTask(random_goal_maze.NullGoalMaze):
    # Adapted from dm_control.locomotion.tasks.RepeatSingleGoalMaze

    def __init__(self,
                 walker,
                 maze_arena,
                 n_targets=3,
                 target_radius=0.3,
                 target_height_above_ground=0.0,
                 target_reward_scale=1.0,
                 enable_global_task_observables=False,
                 camera_resolution=64,
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
                height_above_ground=target_radius + target_height_above_ground,
                rgb1=tuple(color * 1.0),
                rgb2=tuple(color * 1.0),
            )
            self._targets.append(target)
            self._maze_arena.attach(target)
        self._current_target_ix = 0
        self._rewarded_this_step = False
        self._targets_obtained = 0

        if enable_global_task_observables:
            # Add egocentric vectors to targets
            xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

            def _target_pos(physics, target):
                return physics.bind(target.geom).xpos

            for i in range(n_targets):
                # Absolute target position
                walker.observables.add_observable(
                    f'target_abs_{i}',
                    observable_lib.Generic(functools.partial(_target_pos, target=self._targets[i])),
                )
                # Relative target position
                walker.observables.add_egocentric_vector(
                    f'target_rel_{i}',
                    observable_lib.Generic(functools.partial(_target_pos, target=self._targets[i])),
                    origin_callable=xpos_origin_callable)

        self._task_observables = super().task_observables

        def _current_target_index(_):
            return self._current_target_ix

        def _current_target_color(_):
            return TARGET_COLORS[self._current_target_ix]

        self._task_observables['target_index'] = observable_lib.Generic(_current_target_index)
        self._task_observables['target_index'].enabled = True
        self._task_observables['target_color'] = observable_lib.Generic(_current_target_color)
        self._task_observables['target_color'].enabled = True

        self._walker.observables.egocentric_camera.height = camera_resolution
        self._walker.observables.egocentric_camera.width = camera_resolution
        self._maze_arena.observables.top_camera.height = camera_resolution
        self._maze_arena.observables.top_camera.width = camera_resolution

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


class FixedWallTexture(labmaze_textures.WallTextures):
    """Selects a single texture instead of a collection to sample from."""

    def _build(self, style, texture_name):
        labmaze_textures = labmaze_assets.get_wall_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        if texture_name not in labmaze_textures:
            raise ValueError(f'`texture_name` should be one of {labmaze_textures.keys()}: got {texture_name}')
        texture_path = labmaze_textures[texture_name]
        self._textures.append(self._mjcf_root.asset.add(  # type: ignore
            'texture', type='2d', name=texture_name,
            file=texture_path.format(texture_name)))


class FixedFloorTexture(labmaze_textures.FloorTextures):
    """Selects a single texture instead of a collection to sample from."""

    def _build(self, style, texture_names):
        labmaze_textures = labmaze_assets.get_floor_texture_paths(style)
        self._mjcf_root = mjcf.RootElement(model='labmaze_' + style)
        self._textures = []
        if isinstance(texture_names, str):
            texture_names = [texture_names]
        for texture_name in texture_names:
            if texture_name not in labmaze_textures:
                raise ValueError(f'`texture_name` should be one of {labmaze_textures.keys()}: got {texture_name}')
            texture_path = labmaze_textures[texture_name]
            self._textures.append(self._mjcf_root.asset.add(  # type: ignore
                'texture', type='2d', name=texture_name,
                file=texture_path.format(texture_name)))


class MazeWithTargetsArena(mazes.MazeWithTargets):
    """Fork of mazes.RandomMazeWithTargets."""

    def _build(self,
               x_cells,
               y_cells,
               xy_scale=2.0,
               z_height=2.0,
               max_rooms=4,
               room_min_size=3,
               room_max_size=5,
               spawns_per_room=0,
               targets_per_room=0,
               max_variations=26,
               simplify=True,
               skybox_texture=None,
               wall_textures=None,
               floor_textures=None,
               aesthetic='default',
               name='random_maze'):
        random_seed = np.random.randint(2147483648)
        super()._build(
            maze=TextMazeVaryingWalls(
                height=y_cells,
                width=x_cells,
                max_rooms=max_rooms,
                room_min_size=room_min_size,
                room_max_size=room_max_size,
                max_variations=max_variations,
                spawns_per_room=spawns_per_room,
                objects_per_room=targets_per_room,
                simplify=simplify,
                random_seed=random_seed),
            xy_scale=xy_scale,
            z_height=z_height,
            skybox_texture=skybox_texture,
            wall_textures=wall_textures,
            floor_textures=floor_textures,
            aesthetic=aesthetic,
            name=name)

    def _make_floor_variations(self, build_tile_geoms_fn=None):
        """Fork of mazes.MazeWithTargets._make_floor_variations().

        Makes the room floors different if possible, instead of sampling randomly.
        """
        _DEFAULT_FLOOR_CHAR = '.'

        main_floor_texture = self._floor_textures[0]
        if len(self._floor_textures) > 1:
            room_floor_textures = self._floor_textures[1:]
        else:
            room_floor_textures = [main_floor_texture]

        for i_var, variation in enumerate(_DEFAULT_FLOOR_CHAR + string.ascii_uppercase):
            if variation not in self._maze.variations_layer:
                break

            if build_tile_geoms_fn is None:
                # Break the floor variation down to odd-sized tiles.
                tiles = covering.make_walls(self._maze.variations_layer,
                                            wall_char=variation,
                                            make_odd_sized_walls=True)
            else:
                tiles = build_tile_geoms_fn(wall_char=variation)

            if variation == _DEFAULT_FLOOR_CHAR:
                variation_texture = main_floor_texture
            else:
                variation_texture = room_floor_textures[i_var % len(room_floor_textures)]

            for i, tile in enumerate(tiles):
                tile_mid = covering.GridCoordinates(
                    (tile.start.y + tile.end.y - 1) / 2,
                    (tile.start.x + tile.end.x - 1) / 2)
                tile_pos = np.array([(tile_mid.x - self._x_offset) * self._xy_scale,
                                     -(tile_mid.y - self._y_offset) * self._xy_scale,
                                     0.0])
                tile_size = np.array([(tile.end.x - tile_mid.x - 0.5) * self._xy_scale,
                                      (tile.end.y - tile_mid.y - 0.5) * self._xy_scale,
                                      self._xy_scale])
                if variation == _DEFAULT_FLOOR_CHAR:
                    tile_name = 'floor_{}'.format(i)
                else:
                    tile_name = 'floor_{}_{}'.format(variation, i)
                self._tile_geom_names[tile.start] = tile_name
                self._texturing_material_names.append(tile_name)
                self._texturing_geom_names.append(tile_name)
                material = self._mjcf_root.asset.add(
                    'material', name=tile_name, texture=variation_texture,
                    texrepeat=(2 * tile_size[[0, 1]] / self._xy_scale))
                self._mjcf_root.worldbody.add(
                    'geom', name=tile_name, type='plane', material=material,
                    pos=tile_pos, size=tile_size, contype=0, conaffinity=0)


class TextMazeVaryingWalls(labmaze.RandomMaze):
    """Augments standard generated labmaze with some walls marked with different chars."""

    def regenerate(self):
        super().regenerate()
        self._block_variations()

    def _block_variations(self):
        nblocks = 3
        wall_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        n = self.entity_layer.shape[0]
        ivar = 0
        for i in range(nblocks):
            for j in range(nblocks):
                i_from = i * n // nblocks
                i_to = (i + 1) * n // nblocks
                j_from = j * n // nblocks
                j_to = (j + 1) * n // nblocks
                self._change_block_char(i_from, i_to, j_from, j_to, wall_chars[ivar])
                ivar += 1

    def _change_block_char(self, i1, i2, j1, j2, char):
        grid = self.entity_layer
        i, j = np.where(grid[i1:i2, j1:j2] == '*')
        grid[i + i1, j + j1] = char
