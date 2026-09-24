"""
Part 3: Train an RL Agent (REINFORCE)
=====================================
Goal: train a small neural network to balance CartPole using the REINFORCE
policy-gradient algorithm.

The main idea:
  - The policy is a neural network that takes the current state (4 numbers)
    and outputs probabilities for each action (push left or push right).
  - The agent plays one complete episode and records every action and reward.
  - After the episode ends, we compute the return for each timestep: the
    total discounted reward from that point onward.
  - We train the policy to choose actions that lead to higher returns more often,
    while making actions that lead to lower returns less likely.

This "collect an episode, then update the policy" loop is the same training
pattern used in larger robotics RL systems such as PPO in Isaac Sim. CartPole
is simply a much smaller environment with one simulated system instead of
thousands running in parallel.

Fill in the 3 TODOs, then run:

    python 03_reinforce_cartpole.py

Training takes a few minutes. The average reward should gradually improve
from around 20 toward 200+ over a few hundred episodes.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)

GAMMA = 0.99 # How much future rewards matter compared to immediate rewards
LEARNING_RATE = 1e-2
NUM_EPISODES = 600


class PolicyNetwork(nn.Module):
    """Takes a 4-number state and outputs a probability for each of 2 actions."""

    def __init__(self, state_dim=4, num_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state):
        logits = self.net(state)

        # Convert the network's raw outputs (logits) into probabilities.
        # The probabilities for all actions add up to 1.
        return torch.softmax(logits, dim=-1)


def compute_returns(rewards, gamma):
    """
   Compute returns from the end of the episode toward the beginning.
   We go backward because the return at timestep t includes the
   discounted return from the next timestep:
  
       return[t] = rewards[t] + gamma * return[t + 1]
  
   At the final timestep, there is no future return, so we start
   the running total at 0.0. Each step then adds the current reward
   to the discounted total accumulated from future timesteps.
  
   For example, for rewards [r0, r1, r2]:
  
       return[2] = r2
       return[1] = r1 + gamma * r2
       return[0] = r0 + gamma * r1 + gamma^2 * r2
   """ 
    # TODO 1: Compute the discounted return for each timestep.
    #
    # Work backward from the end of the episode because the return at
    # timestep `t` depends on the rewards that come after it.
    #
    # 1. Start `running_total` at 0.0.
    # 2. Iterate through `rewards` from the last timestep to the first.
    # 3. At each timestep, add the current reward to the discounted
    #     return accumulated from future timesteps.
    # 4. Store the result in `returns[t]`.
    #
    # The update at each timestep should be:
    #     running_total = rewards[t] + gamma * running_total

    # Hint: `range(len(rewards) - 1, -1, -1)` iterates backward.

    running_total = 0.0
    returns = [0.0] * len(rewards)

    for t in range(len(rewards) - 1, -1, -1):
        running_total = rewards[t] + gamma * running_total
        returns[t] = running_total

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize the returns to make training more stable.
    # This rescales them to have approximately zero mean and unit variance,
    # so unusually large or small return values have less impact on training.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns

def select_action(policy, state):
    """
    Use the policy to choose an action for the current state.

    The policy outputs probabilities for both actions. Instead of always
    choosing the most likely action, we sample from this distribution so
    the agent can explore different actions during training.

    Returns:
        action: the sampled action (0 or 1)
        log_prob: the log-probability of the sampled action, used later
                  to compute the policy loss.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = policy(state_tensor)  # shape: (1, 2), e.g. [[0.7, 0.3]]

    # Create a probability distribution from the policy's output, then
    # sample one action according to those probabilities.
    #
    # TODO 2: Sample an action from the policy's probability distribution.
    #
    # The policy gives us the probability of each action in `probs`.
    # Use `torch.distributions.Categorical` to turn these probabilities
    # into a distribution we can sample from.
    #
    # Then:
    #   1. Create a Categorical distribution using `probs`.
    #   2. Sample one action from that distribution.
    #   3. Get the log-probability of the sampled action.
    #
    # You will need:
    #     distribution = torch.distributions.Categorical(probs)
    #     action = distribution.sample()
    #     log_prob = distribution.log_prob(action)

    distribution = torch.distributions.Categorical(probs)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)

    return action.item(), log_prob


def train():
    env = gym.make("CartPole-v1")

    policy = PolicyNetwork()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for episode in range(NUM_EPISODES):
        # 1. Start a new episode and get the initial state.
        state, info = env.reset()

        # 2. Store information from each timestep so we can update the policy
        # after the episode is finished.
        log_probs = []
        rewards = []

        done = False

        # 3. Run the episode until the environment says it is finished.
        while not done:
            # Use the policy to choose an action.
            action, log_prob = select_action(policy, state)

            # Apply the action and receive the next state and reward.
            state, reward, terminated, truncated, info = env.step(action)

            # The episode can end naturally (`terminated`) or because it
            # reached a time limit (`truncated`).
            done = terminated or truncated

            # Save the action's log-probability and the reward for later.
            log_probs.append(log_prob)
            rewards.append(reward)

        # 4. Compute the discounted return for every timestep in the episode.
        returns = compute_returns(rewards, GAMMA)

        # 5. REINFORCE policy loss
        # We have two values for every timestep:
        #   - `log_probs`: how likely the policy was to choose each action.
        #   - `returns`: how good the outcome was from that timestep onward.
        #
        # For each timestep, the loss is:
        #
        #     loss_t = -log_prob[t] * return[t]
        #
        # A high return encourages the policy to increase the probability
        # of that action. A low return pushes the policy in the opposite
        # direction.
        #
        # The negative sign is used because optimizers minimize loss, while
        # REINFORCE is trying to maximize the expected return.

        # TODO 3: Build the REINFORCE policy loss.
        # Build the loss in three steps:
        # 1. Create an empty list called `losses` to store the loss
        #    calculated for each timestep.
        #
        # 2. Use `zip(log_probs, returns)` to loop through the matching
        #    log-probability and return for each timestep.
        #
        #    For each pair, calculate:
        #
        #        -log_prob * return
        #
        #    and append the result to `losses`.
        #
        # 3. Use `torch.stack(losses)` to combine the individual losses
        #    into one tensor, then use `.sum()` to get the total
        #    `policy_loss` for the episode.

        losses = []
        for log_prob, return_value in zip(log_probs, returns):
            losses.append(-log_prob * return_value)
            policy_loss = torch.stack(losses).sum()

        # 6. Clear gradients from the previous training step.
        optimizer.zero_grad()

        # 7. Compute how each model parameter contributed to the policy loss.
        policy_loss.backward()

        # 8. Update the model parameters using those gradients.
        optimizer.step()

        # 9. Track the total reward collected during this episode.
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Print the average reward every 25 episodes to monitor progress.
        if (episode + 1) % 25 == 0:
            avg_last_25 = sum(episode_rewards[-25:]) / 25
            print(
                f"episode {episode + 1:4d}/{NUM_EPISODES}  "
                f"avg reward (last 25) = {avg_last_25:.1f}"
            )

    env.close()

    return episode_rewards


if __name__ == "__main__":
    rewards = train()

    # Plot rewards from each episode and a rolling average to show the
    # overall training trend.
    plt.figure(figsize=(7, 4))
    plt.plot(rewards, alpha=0.3, label="per episode")

    # A rolling average smooths out noisy episode rewards and makes it
    # easier to see whether the agent is improving over time.
    window = 25

    if len(rewards) >= window:
        rolling = [
            sum(rewards[i - window:i]) / window
            for i in range(window, len(rewards))
        ]

        plt.plot(
            range(window, len(rewards)),
            rolling,
            label=f"{window}-episode average",
        )

    plt.axhline(
        500,
        color="green",
        linestyle="--",
        label="max possible (500)",
    )

    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("REINFORCE on CartPole-v1")
    plt.legend()
    plt.tight_layout()

    plt.savefig("training_curve.png")

    print("\nSaved plot to training_curve.png")

    plt.show()
