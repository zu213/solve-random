import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import game.rngs as rngs

# Hyperparameters
learning_rate = 0.0005 # if too low x if too hig y
discount_factor = 0.97 # if too low x if too high y
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.95
batch_size = 32 # aa
replay_memory_size = 2000 #dss
max_episodes = 500
guess_range = 100  # Range of guesses for game
update_target_every = 10 #dsadss

replay_memory = []

def create_model():
    model = keras.Sequential([
        layers.LSTM(32, input_shape=(1, 1), return_sequences=True),
        layers.LSTM(32),
        layers.Dense(64, activation='relu'),
        layers.Dense(guess_range, activation='linear')  # Q-values for each possible guess
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Initialize the Q-network and the target network
model = create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

def choose_action(state, action_counts, total_actions):
    if np.random.rand() <= epsilon:
        return random.randint(1, guess_range)
    
    q_values = model.predict(np.array([[state]]).reshape(1, 1, 1), verbose=0)
    ucb_values = q_values + 2 * np.sqrt(np.log(total_actions + 1) / (np.array(action_counts) + 1))
    return np.argmax(ucb_values) + 1

def store_experience(state, action, reward, next_state):
    replay_memory.append((state, action, reward, next_state))
    if len(replay_memory) > replay_memory_size:
        replay_memory.pop(0)

def train_model():
    if len(replay_memory) < batch_size:
        return
    
    batch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    
    states = np.array(states).reshape(-1, 1, 1)
    next_states = np.array(next_states).reshape(-1, 1, 1)
    
    target_q_values = target_model.predict(next_states, verbose=0)
    target = model.predict(states, verbose=0)
    
    for i in range(batch_size):
        if rewards[i] == 1:
            target[i][actions[i] - 1] = rewards[i]
        else:
            target[i][actions[i] - 1] = rewards[i] + discount_factor * np.max(target_q_values[i])
    
    model.fit(states, target, epochs=1, verbose=0)

def update_target_network():
    target_model.set_weights(model.get_weights())

def game_loop(game_func):
    global epsilon
    rewards = []
    action_counts = np.zeros(guess_range)
    total_actions = 1
    
    for episode in range(max_episodes):
        state = random.randint(1, guess_range)
        total_reward = 0
        counter = 0
        while True:
            counter += 1
            action = choose_action(state, action_counts, total_actions)
            action_counts[action - 1] += 1
            total_actions += 1
            
            output = game_func(state)
            reward = 1 if action == output else -1
            
            store_experience(state, action, reward, output)
            train_model()
            total_reward += reward
            
            if reward == 1:
                print(f"Episode {episode+1}: Correct Guess! The number was {output}. Took {counter} guesses")
                break
            
            state = output
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        rewards.append(total_reward)
        
        if episode % update_target_every == 0:
            update_target_network()
    
    plt.plot(range(max_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

game_loop(rngs.not_random_multiplication)
