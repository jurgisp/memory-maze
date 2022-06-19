# dmc-memory-maze

Memory Maze environment for RL based on [dm_control / MuJoCo](https://github.com/deepmind/dm_control).

## Task

Memory Maze is a task designed to test the memory abilities of RL agents.

The task is based on a game known as Scavenger Hunt (or Treasure Hunt). The agent starts in a randomly generated maze, which contains a number of landmarks of different colors. Agent is prompted to find the target landmark of a specific color, indicated by the border color in the observation image. Once the agent successfully finds and touches the correct landmark, it gets a +1 reward and the next random landmark is chosen as a target. If the agent touches the landmark of the wrong color, there is no effect. Throughout the episode the maze layout and the locations of the landmarks do not change. The episode continues for a fixed amount of time, and so the total episode reward is equal to the number of targets the agent can find in the given time. 

[TODO: gif]

Memory Maze tests the memory of the agent in a clean and direct way, because an agent with perfect memory will only have to explore the maze once (which is possible in a time much shorter than the length of episode) and then just follow the shortest path to the target, whereas an agent with no memory will have to randomly wonder through the maze to find each target.

There are 4 size variations of the maze. The largest maze 15x15 is designed to be challenging but solvable for humans (see benchmark results below), but out of reach for the state-of-the-art RL methods. The smaller sizes are provided as stepping stones, with 9x9 solvable with current RL methods.

[TODO: table]

[TODO: top-down images]

## Installation

The environment is available as a pip package
```
pip install git+https://github.com/jurgisp/dmc-memory-maze.git#egg=dmc-memory-maze
```
It will automatically install [`dm_control`](https://github.com/deepmind/dm_control) and [`mujoco`](https://github.com/deepmind/mujoco) dependencies.

## Gym interface

Once pip package is installed, the environment can be created using [Gym](https://github.com/openai/gym) interface

```python
import gym

env = gym.make('dmc_memory_maze:MemoryMaze-9x9-v0')
env = gym.make('dmc_memory_maze:MemoryMaze-11x11-v0')
env = gym.make('dmc_memory_maze:MemoryMaze-13x13-v0')
env = gym.make('dmc_memory_maze:MemoryMaze-15x15-v0')
```

This default environment has dictionary observation space (TODO: map, targets)
```
>>> env.observation_space
Dict(image: Box(0, 255, (64, 64, 3), uint8))
```

In order to make an environment with pure image observation, which may be expected by default RL implementations, add the `-Img-v0` suffix to the env id:
```python
env = gym.make('dmc_memory_maze:MemoryMaze-9x9-Img-v0')
```

There are other helper variations of the environment, see [here](dmc_memory_maze/__init__.py).

## dm_env interface

We also provide [dm_env](https://github.com/deepmind/dm_env) API implementation:

```python
from dmc_memory_maze import tasks

env = tasks.memory_maze_9x9()
env = tasks.memory_maze_11x11()
env = tasks.memory_maze_13x13()
env = tasks.memory_maze_15x15()
```

The observation is a dictionary, which includes image observation (TODO: map, targets)
```python
>>> env.observation_spec()
{
  'image': BoundedArray(shape=(64, 64, 3), ...)
}
```

The constructor accepts a number of arguments, which can be used to tweak the environment for debugging:
```python
env = tasks.memory_maze_9x9(
    control_freq=4,
    discrete_actions=True,
    target_color_in_image=True,
    image_only_obs=False,
    top_camera=False,
    good_visibility=False,
    camera_resolution=64
)
```

## GUI

There is also a graphical UI provided, which can be launched as:

```bash
pip install pygame pillow

# The default view, that the agent sees
python gui/run_gui.py --fps=6 --env "dmc_memory_maze:MemoryMaze-15x15-v0"

# Higher resolution and higher control frequency, nicer for human control
python gui/run_gui.py --fps=60 --env "dmc_memory_maze:MemoryMaze-15x15-HiFreq-HD-v0"
```

## Observation space, Action space

## Benchmarks

### Oracle scores

### Human scores