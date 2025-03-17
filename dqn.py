import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import rngs

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 32
replay_memory_size = 2000
max_episodes = 1000
guess_range = 100  # Range of guesses for game

replay_memory = []

def create_model():
    model = keras.Sequential([
        layers.Dense(64, input_dim=1, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(guess_range, activation='linear')  # Output layer for each action (next guess)
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss='mse')
    return model

# Initialize the Q-network and the target network
model = create_model()
target_model = create_model()

# Copy the weights
target_model.set_weights(model.get_weights())

# Function to choose an action (next guess) using epsilon-greedy strategy
def choose_action(state):
    # There is a chance the guess is random for exploration
    if np.random.rand() <= epsilon:
        return random.randint(1, guess_range)
    # Else we use the model to pick the 'best' option
    q_values = model.predict(np.array([state]))  # Predict Q-values for state
    return np.argmax(q_values) + 1  # Action corresponds to the guess

# Function to store experiences in replay memory
def store_experience(state, action, reward, next_state):
    replay_memory.append((state, action, reward, next_state))
    if len(replay_memory) > replay_memory_size:
        replay_memory.pop(0)

def train_model():
    # Fill up the replay memory
    if len(replay_memory) < batch_size:
        return
    
    batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    
    # Prepare training data
    states = np.array(states)
    next_states = np.array(next_states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Get the Q-values for the next states from the target network
    target_q_values = target_model.predict(next_states)
    
    # Update the Q-values for the chosen actions
    for i in range(batch_size):
        if rewards[i] == 1:
            target_q_values[i][actions[i] - 1] = rewards[i]  # Correct guess, set Q-value to 1
        else:
            target_q_values[i][actions[i] - 1] = rewards[i] + discount_factor * np.max(target_q_values[i])
    
    # Update the model
    model.fit(states, target_q_values, epochs=1, verbose=0)

# Function to update the target network periodically
def update_target_network():
    target_model.set_weights(model.get_weights())

# Game loop
def game_loop(game_func):
    global epsilon
    rewards = []
    
    for episode in range(max_episodes):
        state = random.randint(1, guess_range)  # Random start
        total_reward = 0
        while True:
            action = choose_action(state)  # Choose the next guess
            output = game_func(state)  # Get the correct answer (from game_func)
            reward = 1 if action == output else -0.1  # Reward: +1 if correct, 0 if wrong
            
            store_experience(state, action, reward, output)
            train_model()
            total_reward += reward
            
            # Check if the guess is correct
            if reward == 1:
                print(f"Episode {episode+1}: Correct Guess! The number was {output}.")
                break
            
            # Update the state for the next iteration (next guess)
            state = output

        # Update epsilon (decay it over time)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        rewards.append(total_reward)
        # Update target network periodically
        if episode % 10 == 0:
            update_target_network()

    # Plot the reward over episodes
    plt.plot(range(max_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Start the game loop
game_loop(rngs.not_random_addition)
