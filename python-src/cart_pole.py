import gymnasium as gym
import sys
import torch
from torch import nn
from dqn import DoubleDeepAgent, generate_policy
from epsilon_greedy import EpsilonGreedy, EpsilonUpdateStrategy
from experience_buffer import RandomExperienceBuffer
from trainer import Trainer, TrainResults


def main():
    args = sys.argv
    seed = int(args[1])
    verbose = int(args[2])
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_env = gym.make("CartPole-v1")
    train_env.reset(seed=seed)
    eval_env = gym.make("CartPole-v1")
    eval_env.reset(seed=seed + 1)

    update_strategy = EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing(
        start=0.5, end=0.05, end_fraction=0.2
    )
    action_selector = EpsilonGreedy(0.5, seed + 2, update_strategy)

    mem_replay = RandomExperienceBuffer(10_000, 4, 1_000, seed + 3, False, device)

    policy = generate_policy(
        [
            (256, nn.ReLU()),
            (256, nn.ReLU()),
        ],
        nn.Identity(),
        4,
        2,
    )

    optimizer = torch.optim.Adam
    loss_fn = nn.HuberLoss()

    model = DoubleDeepAgent(
        action_selector,
        mem_replay,
        policy,
        optimizer,
        loss_fn,
        0.01,
        0.99,
        10.0,
        device,
    )

    trainer = Trainer(train_env, eval_env)
    trainer.early_stop = lambda reward: reward >= 475.0

    training_results = trainer.train_by_steps(model, 50_000, 32, 64, 256, 10, 1000, 10, verbose)
    training_steps = sum(training_results[1])

    evaluation_results = trainer.evaluate(model, 10)
    rewards = evaluation_results[0]
    reward_avg = sum(rewards) / len(rewards)
    variance = sum((reward_avg - value) ** 2 for value in rewards) / len(rewards)
    std = variance**0.5

    print(f"python,{seed},{training_steps},{reward_avg},{std}")


if __name__ == "__main__":
    main()