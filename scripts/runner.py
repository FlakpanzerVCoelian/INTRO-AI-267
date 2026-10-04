"""This runner utility is linked to the Vacuum World environment"""
from scripts import *
from scripts.agent import Agent

import gymnasium as gym

def run_episode(env: gym.Env, agent: Agent, render: bool = True) -> float:
    """Run one episode and return the accumulated reward."""

    observation, info = env.reset()

    if render:
        env.render()

    total_reward = 0.0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        
        action = agent.act(observation)

        observation, reward, terminated, truncated, info = env.step(action)

        if render:
            env.render()

        total_reward += reward

    return total_reward

def human_testing(env: gym.Env):
    """Assumes that a human player is playing, so env.render_mode = RenderMode.HUMAN"""
    assert env.render_mode == RenderMode.HUMAN
    import pygame

    done = False
    truncated = False
    current_action = 0

    while not (done or truncated):

        current_action = 4

        if env.window is not None:
            for event in pygame.event.get(pygame.KEYDOWN):
                if event.key == pygame.K_w:
                    current_action = 0
                elif event.key == pygame.K_s:
                    current_action = 1
                elif event.key == pygame.K_a:
                    current_action = 2
                elif event.key == pygame.K_d:
                    current_action = 3
                elif event.key == pygame.K_q:
                    current_action = 5
        
        _, _, done, truncated, _ = env.step(current_action)

        env.render()

    env.close()

def evaluate(env, agent):
    pass

def train(env, agent):
    pass

def collect_trajectory(env, agent):
    pass

if __name__ == '__main__':
    
    from scripts.environment import VacuumWorld
    from scripts.agent import RandomAgent

    env = VacuumWorld()
    ag = RandomAgent(env)
    run_episode(env, ag)