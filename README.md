# dmc-memory-maze

Memory Maze environment for RL based on [dm_control](https://github.com/deepmind/dm_control).

## Task

Memory Maze is a task designed to test the memory abilities of RL agents.

The task is based on a game known as Scavenger Hunt (or Treasure Hunt). The agent starts in a randomly generated maze, which contains a number of landmarks of different colors. Agent is prompted to find the target landmark of a specific color, indicated by the border color in the observation image. Once the agent successfully finds and touches the correct landmark, it gets a +1 reward and the next random landmark is chosen as a target. If the agent touches the landmark of the wrong color, there is no effect. Throughout the episode the maze layout and the locations of the landmarks do not change. The episode continues for a fixed amount of time, and so the total episode reward is equal to the number of targets the agent can find in the given time. 

<p align="center">
    <img width="256" src="https://user-images.githubusercontent.com/3135115/177040240-847f0f0d-b20b-4652-83c3-a486f6f22c22.gif">
</p>

Memory Maze tests the memory of the agent in a clean and direct way, because an agent with perfect memory will only have to explore the maze once (which is possible in a time much shorter than the length of episode) and then just follow the shortest path to the target, whereas an agent with no memory will have to randomly wonder through the maze to find each target.

There are 4 size variations of the maze. The largest maze 15x15 is designed to be challenging but solvable for humans (see benchmark results below), but out of reach for the state-of-the-art RL methods. The smaller sizes are provided as stepping stones, with 9x9 solvable with current RL methods.

| Size      | Landmarks | Episode steps | env_id                |
|-----------|-----------|---------------|-----------------------|
| **9x9**   | 3         | 1000          | `MemoryMaze-9x9-v0`   |
| **11x11** | 4         | 2000          | `MemoryMaze-11x11-v0` |
| **13x13** | 5         | 3000          | `MemoryMaze-13x13-v0` |
| **15x15** | 6         | 4000          | `MemoryMaze-15x15-v0` |

Note that the mazes are generated with [labmaze](https://github.com/deepmind/labmaze), the same algorithm as used by [DmLab-30](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30). In particular, 9x9 corresponds to the [small](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30#goal-locations-small) variant and 15x15 corresponds to the [large](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30#goal-locations-large) variant.

<p align="center">
    <img width="20%" alt="map-9x9" src="https://user-images.githubusercontent.com/3135115/177040204-fbf3b558-d063-49d3-9973-ae113137782f.png">
    &nbsp;
    <img width="20%" alt="map-11x11" src="https://user-images.githubusercontent.com/3135115/177040184-16ccb614-b897-44db-ab2c-7ae66e14c007.png">
    &nbsp;
    <img width="20%" alt="map-13x13" src="https://user-images.githubusercontent.com/3135115/177040164-d3edb11f-de6a-4c17-bce2-38e539639f40.png">
    &nbsp;
    <img width="20%" alt="map-15x15" src="https://user-images.githubusercontent.com/3135115/177040126-b9a0f861-b15b-492c-9216-89502e8f8ae9.png">
    <br/>
    Examples of generated mazes for 4 different sizes.
</p>

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
```python
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
