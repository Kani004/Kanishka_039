import numpy as np
import gym

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Set Q-learning parameters
learning_rate = 0.8      # Alpha
discount_factor = 0.95   # Gamma
epsilon = 1.0            # Exploration rate
epsilon_decay = 0.99
num_episodes = 10000
max_steps = 100

# Initialize the Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    step = 0
    
    # Decay epsilon for exploration-exploitation trade-off
    epsilon = max(0.1, epsilon * epsilon_decay)

    for step in range(max_steps):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action and observe the outcome
        next_state, reward, done, _ = env.step(action)

        # Update Q-value
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        q_table[state, action] = new_value

        # Move to the next state
        state = next_state

        # If the episode is done, break the loop
        if done:
            break

print("Training finished!\n")
print("Q-table values:")
print(q_table)

# Test the learned policy
state = env.reset()
env.render()
total_reward = 0
for _ in range(max_steps):
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    if done:
        break

print("Total Reward in Test:", total_reward)
