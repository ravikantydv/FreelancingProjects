#!/usr/bin/env python
# coding: utf-8

# ### Project 2: Deep Q-Learning for Grid World

# In[1]:


# Importing the required libraries

import numpy as np
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# ### Step 1: Design Your Grid World
# Let's design a new grid world problem with the following elements:
# 
# A 6x6 grid world
# 
# 'S' - Start position
# 
# 'G' - Goal position
# 
# '#' - Obstacle
# 
# 'O' - Open path
S - O O O - 
- # - O G -
- O - # - -
- # O O - -
- O O O O O
- - - - - O
# ### Step 2: Define States, Actions, and Rewards
# States: Each cell in the grid represents a state. We can represent states as coordinates (row, column).
# 
# Actions: The agent can take four actions: move up, move down, move left, and move right.
# 
# Rewards:--
# 
# Reaching the goal: +10
# 
# Hitting an obstacle: -10
# 
# Moving to an open cell: -1

# ### Step 3: Design and Implement the Deep Q-Network

# In[2]:


grid_world = np.array([
    ['S', 'O', 'O', 'O', '-', '-'],
    ['-', '#', '-', 'O', 'G', '-'],
    ['-', 'O', '-', '#', '-', '-'],
    ['-', '#', 'O', 'O', '-', '-'],
    ['-', 'O', 'O', 'O', 'O', 'O'],
    ['-', '-', '-', '-', '-', 'O']
])

# Define rewards
rewards = {
    'G': 10,   # Goal
    '#': -10,  # Obstacle
    'O': -1,   # Open path
    '-': -1,   # Empty cell (not used)
    'S': 0     # Start (no immediate reward)
}


# In[3]:


# Actions (up, down, left, right)


actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# In[4]:


# Define Q-learning parameters
epsilon = 0.2
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 10


# In[5]:


# Initialize the Q-table with zeros

num_rows, num_cols = grid_world.shape
num_actions = len(actions)
q_table = np.zeros((num_rows, num_cols, num_actions))


# In[6]:


# Define the deep Q-network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_rows, num_cols, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_actions)  # Output layer with one neuron per action
])


# In[7]:


# compile the model--

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')


# In[8]:


# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# In[9]:


# Training parameters
num_episodes = 10
batch_size = 32
epsilon = 0.1
discount_factor = 0.9


# In[10]:


def grid_world_to_input(state):
    input_grid = np.zeros((num_rows, num_cols, 1))
    input_grid[state[0], state[1]] = 1
    return input_grid


# In[11]:


# Initialize lists to store MSE and weight trajectories
mse_values = []
weight_trajectories = []

for episode in range(num_episodes):
    state = (0, 0)
    done = False
    total_reward = 0

    while not done:
        # Choose an action using epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(num_actions)
        else:
            # Use the deep Q-network to predict Q-values for each action
            q_values = model.predict(np.array([grid_world_to_input(state)])[..., np.newaxis])
            action = np.argmax(q_values)

        # Calculate the next state and reward
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

        # Check if the next state is within the grid boundaries
        if 0 <= next_state[0] < num_rows and 0 <= next_state[1] < num_cols:
            reward = rewards[grid_world[next_state[0], next_state[1]]]

            # Use the deep Q-network to predict Q-values for the next state
            next_q_values = model.predict(np.array([grid_world_to_input(next_state)])[..., np.newaxis])

            # Update the Q-value of the chosen action using the Q-learning formula
            q_values[0][action] = reward + discount_factor * np.max(next_q_values)

            # Update the model with the new Q-values
            model.fit(np.array([grid_world_to_input(state)])[..., np.newaxis], np.array([q_values]), epochs=1, verbose=0)

            # Update the current state and total reward
            state = next_state
            total_reward += reward

            # Check if the goal is reached
            if grid_world[state[0], state[1]] == 'G':
                done = True
        else:
            # The next state is out of bounds; choose a different action
            pass
        
    # Calculate the MSE
    mse = np.mean(np.square(np.array(mse_values)))
    mse_values.append(mse)

    # Store the weights of the Q-network
    weights = [layer.get_weights() for layer in model.layers]
    weight_trajectories.append(weights)


# In[12]:


# Define the number of rows and columns in your grid world
num_rows = 6
num_cols = 6

# Calculate the total number of states in the grid world
num_states = num_rows * num_cols


# ### For one episode only --

# In[19]:


for i, episode_weights in enumerate(weight_trajectories):
    for i, layer_weights in enumerate(episode_weights):
        for j in range(len(layer_weights)):
            plt.figure()
            valid_weights = [w for w in episode_weights if len(w) > i]
            if not valid_weights:
                continue
            plt.plot(range(len(valid_weights)), [w[i][j] for w in valid_weights], label=f'Episode {i}')
            plt.xlabel('Episodes')
            plt.ylabel(f'Layer {i}, Weight {j}')
            plt.title(f'Weight Trajectory for Layer {i}, Weight {j}')
            plt.legend()
            plt.show()


# ### Checked for complete 10 episodes

# In[20]:


# Visualization of weight trajectories for multiple episodes
for episode_id, episode_weights in enumerate(weight_trajectories):
    for layer_id, layer_weights in enumerate(episode_weights):
        if layer_id < len(layer_weights):
            for weight_id in range(len(layer_weights[layer_id])):
                plt.figure()
                valid_weights = [w[layer_id][weight_id] for w in episode_weights if len(w) > layer_id and weight_id < len(w[layer_id])]
                if valid_weights:
                    plt.plot(range(len(valid_weights)), valid_weights, label=f'Episode {episode_id}')
                    plt.xlabel('Episodes')
                    plt.ylabel(f'Layer {layer_id}, Weight {weight_id}')
                    plt.title(f'Weight Trajectory for Layer {layer_id}, Weight {weight_id}')
                    plt.legend()
                    plt.show()


# ### Weight trajectories for 10 episodes

# In[21]:


# Visualization of weight trajectories for multiple episodes
for layer_id in range(len(weight_trajectories[0])):
    for weight_id in range(len(weight_trajectories[0][layer_id])):
        plt.figure()
        for episode_id, episode_weights in enumerate(weight_trajectories):
            valid_weights = [w[layer_id][weight_id] for w in episode_weights if len(w) > layer_id and weight_id < len(w[layer_id])]
            if valid_weights:
                plt.plot(range(len(valid_weights)), valid_weights, label=f'Episode {episode_id}')
        plt.xlabel('Episodes')
        plt.ylabel(f'Layer {layer_id}, Weight {weight_id}')
        plt.title(f'Weight Trajectory for Layer {layer_id}, Weight {weight_id}')
        plt.legend()
        plt.show()


# ### Updated weights for 10 episodes--
# 
# Uncomment the code if you want to check the updated weights (Note: It will take some time to run)

# In[3]:


# # Visualization of weight trajectories for multiple episodes
# for layer_id in range(len(weight_trajectories[0])):
#     for weight_id in range(len(weight_trajectories[0][layer_id])):
#         plt.figure()
#         for episode_id, episode_weights in enumerate(weight_trajectories):
#             valid_weights = [w[layer_id][weight_id] for w in episode_weights if len(w) > layer_id and weight_id < len(w[layer_id])]
#             if valid_weights:
#                 plt.plot(range(len(valid_weights)), valid_weights, label=f'Episode {episode_id}')
#         plt.xlabel('Episodes')
#         plt.ylabel(f'Layer {layer_id}, Weight {weight_id}')
#         plt.title(f'Weight Trajectory for Layer {layer_id}, Weight {weight_id}')
        
#         # Add legend if there are valid weights for this layer and weight
#         if any(valid_weights):
#             plt.legend()
        
#         plt.show()


# ### Testing Convergence Process of MSE

# In[6]:


# Plot the MSE convergence after the training loop
plt.figure(figsize=(10, 5))
plt.plot(range(num_episodes), mse_values, label='MSE')
plt.xlabel('Episodes')
plt.ylabel('Mean Square Error')
plt.title('Convergence of Mean Square Error')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




