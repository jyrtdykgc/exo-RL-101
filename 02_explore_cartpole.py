"""
Part 2: Meet a Gymnasium Environment
====================================
No TODOs in this file. Run it first and read the output.

    python 02_explore_cartpole.py

Every Gymnasium environment follows the same interaction loop:

    observation, info = env.reset()
    while not done:
        action = <choose an action>
        observation, reward, terminated, truncated, info = env.step(action)

You'll use this exact pattern later for training RL agents.

CartPole is a simple control task:
- A pole is attached to a cart.
- The agent can push the cart left or right.
- It receives +1 reward for every timestep the pole stays balanced.
- The episode ends if the pole falls too far or after 500 timesteps.

This script does **not** use a trained agent. It chooses random actions,
so it serves as a baseline for comparison before training in Part 3.
"""

import gymnasium as gym

# Create the CartPole environment.
env = gym.make("CartPole-v1")

# The observation is the state the agent receives at every timestep.
print("Observation space:", env.observation_space)
print("  -> 4 numbers: cart position, cart velocity, pole angle, pole angular velocity")

# The action space defines what the agent is allowed to do.
print("Action space:", env.action_space)
print("  -> 2 actions: 0 = push left, 1 = push right")
print()

# Run a few episodes using completely random actions.
for episode in range(5):
    # Reset starts a new episode and returns the initial observation.
    observation, info = env.reset()

    total_reward = 0
    done = False
    steps = 0

    # Keep interacting with the environment until the episode ends.
    while not done:
        # Sample a random valid action from the action space.
        action = env.action_space.sample()

        # Apply the action and receive the next state and reward.
        observation, reward, terminated, truncated, info = env.step(action)

        # The episode ends if it terminates naturally or is truncated.
        done = terminated or truncated

        total_reward += reward
        steps += 1

    print(f"Episode {episode + 1}: survived {steps} steps, total reward = {total_reward}")

# Always close the environment when finished.
env.close()

print(
    "\nA well-trained agent can survive the full 500-step episode almost every time. "
    "A random agent usually lasts only about 10-30 steps."
)

