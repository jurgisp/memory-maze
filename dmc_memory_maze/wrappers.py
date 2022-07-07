

from typing import Any, Dict, List

import dm_env
import numpy as np
from dm_env import specs


class Wrapper(dm_env.Environment):
    """Base class for dm_env.Environment wrapper."""

    def __init__(self, env: dm_env.Environment):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(f'Attempted to get missing private attribute {name}')
        return getattr(self.env, name)

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(action)

    def reset(self) -> dm_env.TimeStep:
        return self.env.reset()

    def action_spec(self) -> Any:
        return self.env.action_spec()

    def discount_spec(self) -> Any:
        return self.env.discount_spec()

    def observation_spec(self) -> Any:
        return self.env.observation_spec()

    def reward_spec(self) -> Any:
        return self.env.reward_spec()

    def close(self):
        return self.env.close()


class ObservationWrapper(Wrapper):
    """Base class for observation wrapper."""

    def observation_spec(self):
        raise NotImplementedError

    def observation(self, obs: Any) -> Any:
        raise NotImplementedError

    def step(self, action) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.step(action)
        return dm_env.TimeStep(step_type, discount, reward, self.observation(observation))

    def reset(self) -> dm_env.TimeStep:
        step_type, discount, reward, observation = self.env.reset()
        return dm_env.TimeStep(step_type, discount, reward, self.observation(observation))


class RemapObservationWrapper(ObservationWrapper):
    """Select a subset of dictionary observation keys and rename them."""

    def __init__(self, env: dm_env.Environment, mapping: Dict[str, str]):
        super().__init__(env)
        self.mapping = mapping

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return {key: spec[key_orig] for key, key_orig in self.mapping.items()}

    def observation(self, obs):
        assert isinstance(obs, dict)
        return {key: obs[key_orig] for key, key_orig in self.mapping.items()}


class TargetsVectorWrapper(ObservationWrapper):
    """Collects and postporcesses walker/target_{i} vectors into targets_vector (n_targets,2) tensor,
    which indicates the relative position of all targets."""

    def __init__(self, env: dm_env.Environment, key='targets_vector', xy_scale=2.0):
        super().__init__(env)
        self.key = key
        self.xy_scale = xy_scale
        spec = self.env.observation_spec()
        assert isinstance(spec, dict) and 'walker/target_0' in spec
        i = 0
        while f'walker/target_{i}' in spec:
            i += 1
        self.n_targets = i

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        spec[self.key] = specs.Array((self.n_targets, 2), float, self.key)
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        x = np.zeros((self.n_targets, 2))
        for i in range(self.n_targets):
            x_raw = obs[f'walker/target_{i}']
            x[i] = x_raw[:2] / self.xy_scale
        obs[self.key] = x
        return obs


class AbsolutePositionWrapper(ObservationWrapper):
    """Postprocesses absolute_position and absolute_orientation."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        # Change absolute_position from 3-vector to 2-vector
        assert 'absolute_position' in spec
        spec['absolute_position'] = specs.Array((2, ), float, 'absolute_position')
        # Change absolute_orientation from 3x3 matrix to 2-vector
        assert 'absolute_orientation' in spec
        spec['absolute_orientation'] = specs.Array((2, ), float, 'absolute_orientation')
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        obs['absolute_position'] = obs['absolute_position'][:2]
        # Pick orientation vector such, that going forward increases absolute_position in the direction of absolute_orientation.
        obs['absolute_orientation'] = obs['absolute_orientation'][:2, 1]  
        return obs


class ImageOnlyObservationWrapper(ObservationWrapper):
    """Select one of the dictionary observation keys as observation."""

    def __init__(self, env: dm_env.Environment, key: str = 'image'):
        super().__init__(env)
        self.key = key

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        return spec[self.key]

    def observation(self, obs):
        assert isinstance(obs, dict)
        return obs[self.key]


class DiscreteActionSetWrapper(Wrapper):
    """Change action space from continuous to discrete with given set of action vectors."""

    def __init__(self, env: dm_env.Environment, action_set: List[np.ndarray]):
        super().__init__(env)
        self.action_set = action_set

    def action_spec(self):
        return specs.DiscreteArray(len(self.action_set))

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(self.action_set[action])


class TargetColorAsBorderWrapper(ObservationWrapper):
    """MemoryMaze-specific wrapper, which draws target_color as border on the image."""

    def observation_spec(self):
        spec = self.env.observation_spec()
        assert isinstance(spec, dict)
        assert 'target_color' in spec
        spec.pop('target_color')
        return spec

    def observation(self, obs):
        assert isinstance(obs, dict)
        assert 'target_color' in obs and 'image' in obs
        target_color = obs.pop('target_color')
        img = obs['image']
        B = int(2 * np.sqrt(img.shape[0] // 64))
        img[:, :B] = target_color * 255 * 0.7
        img[:, -B:] = target_color * 255 * 0.7
        img[:B, :] = target_color * 255 * 0.7
        img[-B:, :] = target_color * 255 * 0.7
        return obs
