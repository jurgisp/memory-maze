
import gym

import argparse
from collections import defaultdict

import numpy as np
import pygame
import pygame.freetype
from PIL import Image

PANEL_LEFT = 250
PANEL_RIGHT = 250
FULL_SCREEN = True   # Full screen to force focus

K_NONE = 0
HOLD_ACTION = True

def get_keymap(env):
    return {
        K_NONE: 0,
        pygame.K_UP: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_DOWN: 4,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='dmc_memory_maze:MemMaze-9x9-v0')
    parser.add_argument('--size', type=int, nargs=2, default=(600, 600))
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--random', type=float, default=0.0)
    parser.add_argument('--noreset', action='store_true')
    parser.add_argument('--record', type=str, default=None)
    args = parser.parse_args()
    render_size = args.size
    window_size = (render_size[0] + PANEL_LEFT + PANEL_RIGHT, render_size[1])

    print(f'Creating environment: {args.env}')
    env = gym.make(args.env, disable_env_checker=True)
    # if args.record:
    #     env = Recorder(env, args.record)  # TODO
    
    keymap = get_keymap(env)

    steps = 0
    return_ = 0
    episode = 0
    obs = env.reset()

    pygame.init()
    screen = pygame.display.set_mode(window_size, pygame.FULLSCREEN if FULL_SCREEN else 0)
    # pygame.display.toggle_fullscreen()
    clock = pygame.time.Clock()
    font = pygame.freetype.SysFont('Mono', 16)
    fontsmall = pygame.freetype.SysFont('Mono', 12)
    running = True
    paused = False
    speedup = False

    while running:

        # Rendering

        screen.fill((64, 64, 64))

        # Render image observation
        assert isinstance(obs, dict), 'Expecting dictionary observation with obs["image"]'
        image = obs['image']  # type: ignore
        image = Image.fromarray(image)
        image = image.resize(render_size, resample=0)
        image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (PANEL_LEFT, 0))

        # Render statistics
        lines = obs_to_text(obs, env, steps, return_)
        y = 5
        for line in lines:
            text_surface, rect = font.render(line, (255, 255, 255))
            screen.blit(text_surface, (16, y))
            y += font.size + 2  # type: ignore

        # # Render keymap help
        # lines = keymap_to_text(keymap, actions)
        # y = 5
        # for line in lines:
        #     text_surface, rect = fontsmall.render(line, (255, 255, 255))
        #     screen.blit(text_surface, (render_size[0] + 230 + 16, y))
        #     y += fontsmall.size + 2

        pygame.display.flip()
        clock.tick(args.fps if not speedup else 0)

        # Keyboard input

        action = None
        force_reset = False
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Window close
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Quit
                    running = False
                if event.key == pygame.K_p:  # Pause
                    paused = not paused
                if event.key == pygame.K_BACKSPACE:  # Force reset
                    force_reset = True
                if event.key in keymap.keys():  # Action key
                    # Action key down
                    action = keymap[event.key]
        
        pressed = pygame.key.get_pressed()
        if action is None:
            action = keymap[K_NONE]  # noop
            if HOLD_ACTION:
                for key, act in keymap.items():
                    if pressed[key]:
                        # Action key hold
                        action = act
                        break

        speedup = pressed[pygame.K_TAB]

        if paused:
            continue

        # Environment step

        if args.random:
            if np.random.random() < args.random:
                action = env.action_space.sample()
        
        obs, reward, done, info = env.step(action)
        steps += 1
        return_ += reward

        # Episode end

        if reward or done:
            print(f'reward: {reward}  done: {done}  info: {info}')
        if done or force_reset:
            print(f'Episode done - length: {steps}  return: {return_}')
            obs = env.reset()
            steps = 0
            return_ = 0
            episode += 1

    pygame.quit()

def obs_to_text(obs, env, steps, return_):
    kvs = []
    kvs.append(('## Stats ##', ''))
    kvs.append(('', ''))
    kvs.append(('step', steps))
    kvs.append(('return', return_))
    lines = [f'{k:<15} {v:>5}' for k, v in kvs]
    return lines

# def keymap_to_text(keymap, all_actions, verbose=False):
#     lookup = defaultdict(list)
#     for key, action in keymap.items():
#         if key != K_NONE:
#             if action in all_actions:
#                 lookup[action].append(pygame.key.name(key))
#             elif verbose:
#                 print(f'WARN: keymap to unknown action: {action}')

#     kvs = []
#     kvs.append(('## Commands ##', ''))
#     kvs.append(('', ''))

#     # mapped actions
#     for action in lookup.keys():
#         kvs.append((action, ', '.join(lookup[action])))

#     # unmapped actions
#     for action in all_actions:
#         if action not in lookup and action != 'noop':
#             kvs.append((action, '-'))

#     # special actions
#     kvs.append(('quit', 'esc'))
#     kvs.append(('pause', 'p'))
#     kvs.append(('speed up', 'tab'))

#     lines = [f'{k:<27} {v}' for k, v in kvs]
#     return lines


if __name__ == '__main__':
    main()
