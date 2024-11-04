import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random

# Initialize environment and parameters
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
num_episodes = 500
memory = deque(maxlen=2000)

# Define the DQN model
def build_model():
    model = Sequential([
        Dense(24, input_dim=state_size, activation="relu"),
        Dense(24, activation="relu"),
        Dense(action_size, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss="mse")
    return model

# Initialize the model
model = build_model()

# Function to get action using epsilon-greedy policy
def get_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explore
    q_values = model.predict(state)
    return np.argmax(q_values[0])  # Exploit

# Function to replay experiences and train the model
def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += discount_factor * np.max(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Train the DQN agent
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for time in range(500):
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            print(f"Episode: {episode+1}/{num_episodes}, Score: {time}, Epsilon: {epsilon:.2}")
            break
        replay()

# Testing the trained model
state = env.reset()
state = np.reshape(state, [1, state_size])
for time in range(500):
    env.render()
    action = np.argmax(model.predict(state)[0])
    next_state, _, done, _ = env.step(action)
    state = np.reshape(next_state, [1, state_size])
    if done:
        break
env.close()
