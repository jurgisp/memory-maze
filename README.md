**Status:** Stable release

[![PyPI](https://img.shields.io/pypi/v/memory-maze.svg)](https://pypi.python.org/pypi/memory-maze/#history)

# Memory Maze

Memory Maze is a 3D domain of randomized mazes designed for evaluating the long-term memory abilities of RL agents. Memory Maze isolates long-term memory from confounding challenges, such as exploration, and requires remembering several pieces of information: the positions of objects, the wall layout, and keeping track of agent’s own position.

| Memory 9x9 | Memory 11x11 | Memory 13x13 | Memory 15x15 |
|------------|--------------|--------------|--------------|
| ![map-9x9](https://user-images.githubusercontent.com/3135115/177040204-fbf3b558-d063-49d3-9973-ae113137782f.png) | ![map-11x11](https://user-images.githubusercontent.com/3135115/177040184-16ccb614-b897-44db-ab2c-7ae66e14c007.png) | ![map-13x13](https://user-images.githubusercontent.com/3135115/177040164-d3edb11f-de6a-4c17-bce2-38e539639f40.png) | ![map-15x15](https://user-images.githubusercontent.com/3135115/177040126-b9a0f861-b15b-492c-9216-89502e8f8ae9.png) |

Key features:
- Online RL memory tasks (with baselines)
- Offline dataset for representation learning (with baselines)
- Verified that memory is the key challenge
- Challenging but solvable by human baseline
- Easy installation via a simple pip command
- Available `gym` and `dm_env` interfaces
- Supports headless and hardware rendering
- Interactive GUI for human players
- Hidden state information for probe evaluation

Also see the accompanying research paper: [Evaluating Long-Term Memory in 3D Mazes](https://arxiv.org/abs/2210.13383)

```
@article{pasukonis2022memmaze,
  title={Evaluating Long-Term Memory in 3D Mazes},
  author={Pasukonis, Jurgis and Lillicrap, Timothy and Hafner, Danijar},
  journal={arXiv preprint arXiv:2210.13383},
  year={2022}
}
```

## Installation

Memory Maze builds on the [`dm_control`](https://github.com/deepmind/dm_control) and [`mujoco`](https://github.com/deepmind/mujoco) packages, which are automatically installed as dependencies:

```sh
pip install memory-maze
```

## Play Yourself

Memory Maze allows you to play the levels in human mode. We used this mode for recording the human baseline scores. These are the instructions for launching the GUI:

```sh
# GUI dependencies
pip install gym pygame pillow imageio

# Launch with standard 64x64 resolution
python gui/run_gui.py

# Launch with higher 256x256 resolution
python gui/run_gui.py --env "memory_maze:MemoryMaze-9x9-HD-v0"
```

## Task Description

The task is based on a game known as scavenger hunt or treasure hunt:
- The agent starts in a randomly generated maze, which contains several objects of different colors.
- The agent is prompted to find the target object of a specific color, indicated by the border color in the observation image.
- Once the agent successfully finds and touches the correct object, it gets a +1 reward and the next random object is chosen as a target.
- If the agent touches an object of the wrong color, there is no effect.
- Throughout the episode, the maze layout and the locations of the objects do not change.
- The episode continues for a fixed amount of time, so the total episode reward equals the number of reached targets.

<p align="center"><img width="256" src="https://user-images.githubusercontent.com/3135115/177040240-847f0f0d-b20b-4652-83c3-a486f6f22c22.gif"></p>

An agent with long-term memory only has to explore each maze once (which is possible in a time much shorter than the length of an episode) and can afterwards follow the shortest path to each requested target, whereas an agent with no memory has to randomly wander through the maze to find each target.

There are 4 size variations of the maze. The largest maze 15x15 is designed to be challenging but solvable for humans (see benchmark results below), but out of reach for the state-of-the-art RL methods. The smaller sizes are provided as stepping stones, with 9x9 being solvable with current RL methods.

| Size | env_id | Objects | Episode steps | Mean human score | Mean max score |
|:---------:|-----------------------|:---:|:-----:|:----:|:----:|
| **9x9**   | `MemoryMaze-9x9-v0`   |  3  | 1000  | 26.4 | 34.8 |
| **11x11** | `MemoryMaze-11x11-v0` |  4  | 2000  | 44.3 | 58.0 |
| **13x13** | `MemoryMaze-13x13-v0` |  5  | 3000  | 55.5 | 74.5 |
| **15x15** | `MemoryMaze-15x15-v0` |  6  | 4000  | 67.7 | 87.7 |

The mazes are generated with [labmaze](https://github.com/deepmind/labmaze), the same algorithm as used by [DmLab-30](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30). The 9x9 corresponds to the [small](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30#goal-locations-small) variant and 15x15 corresponds to the [large](https://github.com/deepmind/lab/tree/master/game_scripts/levels/contributed/dmlab30#goal-locations-large) variant.

## Gym Interface

You can create the environment using the [Gym](https://github.com/openai/gym) interface:

```python
!pip install gym
import gym

# Set this if you are getting "Unable to load EGL library" error:
#  os.environ['MUJOCO_GL'] = 'glfw'  

env = gym.make('memory_maze:MemoryMaze-9x9-v0')
env = gym.make('memory_maze:MemoryMaze-11x11-v0')
env = gym.make('memory_maze:MemoryMaze-13x13-v0')
env = gym.make('memory_maze:MemoryMaze-15x15-v0')
```

**Troubleshooting:** if you are getting "Unable to load EGL library error", that is because we enable MuJoCo headless GPU rendering (`MUJOCO_GL=egl`) by default. If you are testing locally on your machine, you can enable windowed rendering instead (`MUJOCO_GL=glfw`). [Read here](https://github.com/deepmind/dm_control#rendering) about the different rendering options. 

The default environment has 64x64 image observations:

```python
>>> env.observation_space
Box(0, 255, (64, 64, 3), uint8)
```

There are 6 discrete actions:

```python
>>> env.action_space
Discrete(6)  # (noop, forward, left, right, forward_left, forward_right)
```

To create an environment with extra observations for debugging and probe analysis, append `ExtraObs` to the names:

```python
>>> env = gym.make('memory_maze:MemoryMaze-9x9-ExtraObs-v0')
>>> env.observation_space
Dict(
    agent_dir: Box(-inf, inf, (2,), float64), 
    agent_pos: Box(-inf, inf, (2,), float64),
    image: Box(0, 255, (64, 64, 3), uint8),
    maze_layout: Box(0, 1, (9, 9), uint8),
    target_color: Box(-inf, inf, (3,), float64),
    target_pos: Box(-inf, inf, (2,), float64),
    target_vec: Box(-inf, inf, (2,), float64),
    targets_pos: Box(-inf, inf, (3, 2), float64),
    targets_vec: Box(-inf, inf, (3, 2), float64)
)
```

We also register [additional variants](memory_maze/__init__.py) of the environment that can be useful in certain scenarios.

## DeepMind Interface

You can create the environment using the [dm_env](https://github.com/deepmind/dm_env) interface:

```python
from memory_maze import tasks

env = tasks.memory_maze_9x9()
env = tasks.memory_maze_11x11()
env = tasks.memory_maze_13x13()
env = tasks.memory_maze_15x15()
```

Each observation is a dictionary that includes `image` key:

```python
>>> env.observation_spec()
{
  'image': BoundedArray(shape=(64, 64, 3), ...)
}
```

The constructor accepts a number of arguments, which can be used to tweak the environment:

```python
env = tasks.memory_maze_9x9(
    global_observables=True,
    image_only_obs=False,
    top_camera=False,
    camera_resolution=64,
    control_freq=4.0,
    discrete_actions=True,
)
```

## Offline Dataset

[**Dataset download here** (~100GB per dataset)](https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB)

We provide two datasets of experience collected from the Memory Maze environment: Memory Maze 9x9 (30M) and Memory Maze 15x15 (30M). Each dataset contains 30 thousand trajectories from Memory Maze 9x9 and 15x15 environments respectively, split into 29k trajectories for training and 1k for evaluation. All trajectories are 1000 steps long, so each dataset has 30M steps total.

The data is generated with a scripted policy that navigates to randomly chosen points in the maze under action noise. This choice of policy was made to generate diverse trajectories that explore the maze effectively and that form spatial loops, which can be important for learning long-term memory. We intentionally avoid recording data with a trained agent to ensure a diverse data distribution and to avoid dataset bias that could favor some methods over others. Because of this, the rewards are quite sparse in the data, occurring on average 1-2 times per trajectory.

Each trajectory is saved as an NPZ file with the following entries available:

| Key            | Shape              | Type   | Description                                   |
|----------------|--------------------|--------|-----------------------------------------------|
| `image`        | (64, 64, 3)        | uint8  | First-person view observation                 |
| `action`       | (6)                | binary | Last action, one-hot encoded                  |
| `reward`       | ()                 | float  | Last reward                                   |
| `maze_layout`  | (9, 9) or (15, 15) | binary | Maze layout (wall / no wall)                  |
| `agent_pos`    | (2)                | float  | Agent position in global coordinates          |
| `agent_dir`    | (2)                | float  | Agent orientation as a unit vector            |
| `targets_pos`  | (3, 2) or (6, 2)   | float  | Object locations in global coordinates        |
| `targets_vec`  | (3, 2) or (6, 2)   | float  | Object locations in agent-centric coordinates |
| `target_pos`   | (2)                | float  | Current target object location, global        |
| `target_vec`   | (2)                | float  | Current target object location, agent-centric |
| `target_color` | (3)                | float  | Current target object color RGB               |

You can load a trajectory using [`np.load()`](https://numpy.org/doc/stable/reference/generated/numpy.load.html) to obtain a dictionary of Numpy arrays as follows:

```python
episode = np.load('trajectory.npz')
episode = {key: episode[key] for key in episode.keys()}

assert episode['image'].shape == (1001, 64, 64, 3)
assert episode['image'].dtype == np.uint8
```

All tensors have a leading time dimension, e.g. `image` tensor has shape (1001, 64, 64, 3). The tensor length is 1001 because there are 1000 steps (actions) in a trajectory, `image[0]` is the observation *before* the first action, and `image[-1]` is the observation *after* the last action.

## Online RL Baselines

In our [research paper](https://arxiv.org/abs/2210.13383), we evaluate the model-free [IMPALA](https://github.com/google-research/seed_rl/tree/master/agents/vtrace) agent and the model-based [Dreamer](https://github.com/jurgisp/pydreamer) agent as baselines.

<p align="center">
  <img width="650" alt="baselines" src="https://user-images.githubusercontent.com/3135115/197349778-74073613-bf6c-449b-b5c2-07adf21030ff.png">
  <br/>
  <img width="650" alt="training" src="https://user-images.githubusercontent.com/3135115/197485498-60560934-2629-47b0-ada8-0484398800d0.png">
</p>

Here are videos of the learned behaviors:

**Memory 9x9 - Dreamer (TBTT)**

https://user-images.githubusercontent.com/3135115/197378287-4e413440-7097-4d11-8627-3d7fac0845f1.mp4

**Memory 9x9 - IMPALA (400M)**

https://user-images.githubusercontent.com/3135115/197378929-7fe3f374-c11c-409a-8a95-03feeb489330.mp4

**Memory 15x15 - Dreamer (TBTT)**

https://user-images.githubusercontent.com/3135115/197378324-fb99b496-dba8-4b00-ad80-2d6e19ba8acd.mp4

**Memory 15x15 - IMPALA (400M)**

https://user-images.githubusercontent.com/3135115/197378936-939e7615-9dad-4765-b0ef-a49c5a38fe28.mp4

## Offline Probing Baselines

Here we visualize probe predictions alongside trajectories of the offline dataset, as explained in [the paper](https://arxiv.org/abs/2210.13383). These trajectories are from the offline dataset, where the agent just navigates to random points in the maze, it does *not* try to collect rewards.

Bottom-left: Object location predictions (x) versus the actual locations (o).

Bottom-right: Wall layout predictions (dark green = true positive, light green = true negative, light red = false positive, dark red = false negative).

**Memory 9x9 Walls Objects - RSSM (TBTT)**

https://user-images.githubusercontent.com/3135115/197379227-775ec5bc-0780-4dcc-b7f1-660bc7cf95f1.mp4

**Memory 9x9 Walls Objects - Supervised oracle**

https://user-images.githubusercontent.com/3135115/197379235-a5ea0388-2718-4035-8bbc-064ecc9ea444.mp4

**Memory 15x15 Walls Objects - RSSM (TBTT)**

https://user-images.githubusercontent.com/3135115/197379245-fb96bd12-6ef5-481e-adc6-f119a39e8e43.mp4

**Memory 15x15 Walls Objects - Supervised oracle**

https://user-images.githubusercontent.com/3135115/197379248-26a8093e-8b54-443c-b154-e33e0383b5e4.mp4

## Questions

Please [open an issue][issues] on Github.

[issues]: https://github.com/jurgisp/memory-maze/issues
