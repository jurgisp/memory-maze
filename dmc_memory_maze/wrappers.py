

from typing import Any, Dict, List

import dm_env
from dm_env.specs import DiscreteArray
import numpy as np


class Wrapper(dm_env.Environment):

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


class DiscreteActionSetWrapper(Wrapper):

    def __init__(self, env: dm_env.Environment, action_set: List[np.ndarray]):
        super().__init__(env)
        self.action_set = action_set

    def action_spec(self):
        return DiscreteArray(len(self.action_set))

    def step(self, action) -> dm_env.TimeStep:
        return self.env.step(self.action_set[action])
