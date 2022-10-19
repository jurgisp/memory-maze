from dm_env.specs import BoundedArray, DiscreteArray
import numpy as np

def sample_spec(space: BoundedArray) -> np.ndarray:
    if isinstance(space, DiscreteArray):
        return np.random.randint(space.num_values, size=space.shape)
    
    if isinstance(space, BoundedArray):
        return np.random.uniform(space.minimum, space.maximum, size=space.shape)
    
    raise NotImplementedError
