from datetime import datetime
from pathlib import Path

import gym
import imageio
import numpy as np

from PIL import Image


class SaveNpzWrapper(gym.Wrapper):

    def __init__(self, env, log_dir, video_fps=30, video_size=256, video_format='mp4'):
        env = ActionRewardResetWrapper(env)
        env = CollectWrapper(env)
        super().__init__(env)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self.video_size = video_size
        self.video_format = video_format

    def step(self, action):
        obs, reward, done, info = self.env.step(action)  # type: ignore
        data = info.get('episode')
        if data:
            ep_id = info['episode_id']
            ep_reward = data['reward'].sum()
            ep_steps = len(data['reward']) - 1
            ep_name = f'{ep_id}-r{ep_reward:.0f}-{ep_steps:04}'
            self._save_npz(data, self.log_dir / f'{ep_name}.npz')
            if self.video_format:
                self._save_video(data, self.log_dir / f'{ep_name}.{self.video_format}')
        return obs, reward, done, info

    def _save_npz(self, data, path):
        with path.open('wb') as f:
            np.savez_compressed(f, **data)
        print(f'Saved {path}', {k: v.shape for k, v in data.items()})
    
    def _save_video(self, data, path):
        writer = imageio.get_writer(path, fps=self.video_fps)
        for frame in data['image']:
            img = Image.fromarray(frame)
            img = img.resize((self.video_size, self.video_size), resample=0)
            writer.append_data(np.array(img))
        writer.close()
        print(f'Saved {path}')


class CollectWrapper(gym.Wrapper):
    """Copied from pydreamer.envs.wrappers."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.episode = []
        self.episode_id = ''

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode.append(obs.copy())
        if done:
            episode = {k: np.array([t[k] for t in self.episode]) for k in self.episode[0]}
            info['episode'] = episode
        info['episode_id'] = self.episode_id
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.episode = [obs.copy()]
        self.episode_id = datetime.now().strftime('%Y%m%dT%H%M%S')
        return obs


class ActionRewardResetWrapper(gym.Wrapper):
    """Copied from pydreamer.envs.wrappers."""

    def __init__(self, env, no_terminal=False):
        super().__init__(env)
        self.env = env
        self.no_terminal = no_terminal
        # Handle environments with one-hot or discrete action, but collect always as one-hot
        self.action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(action, int):
            action_vec = np.zeros(self.action_size)
            action_vec[action] = 1.0
        else:
            assert isinstance(action, np.ndarray) and action.shape == (self.action_size,), "Wrong one-hot action shape"
            action_vec = action
        obs['action'] = action_vec
        obs['reward'] = np.array(reward)
        obs['terminal'] = np.array(False if self.no_terminal or 'TimeLimit.truncated' in info or info.get('time_limit') else done)
        obs['reset'] = np.array(False)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs['action'] = np.zeros(self.action_size)
        obs['reward'] = np.array(0.0)
        obs['terminal'] = np.array(False)
        obs['reset'] = np.array(True)
        return obs
