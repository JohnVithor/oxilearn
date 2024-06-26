import sys
import gymnasium as gym
import os
import torch 
import numpy as np
import random
from oxilearn import DQNAgent


def main(seed, save, verbose):

    env = gym.make('LunarLander-v2')
    agent = DQNAgent([(256, "relu"), (256, "relu")],
        memory_size=5_000,
        min_memory_size=1_000,
        lr=0.0005,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.0005,
        seed=seed
    )
    agent.prepare(env)
    # agent.load("./safetensors")
    env.reset(seed=seed)

    agent.train(env, env.spec.reward_threshold, steps=1_000_000, verbose=verbose)
    
    if save:
        agent.save("./safetensors-rust")

    return agent.evaluate(env, 10)

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.environ['PYTHONASHSEED'] = f'{seed}' 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    reward, std = main(seed, save, verbose)
    print(f"rust,{seed},{reward},{std}")


