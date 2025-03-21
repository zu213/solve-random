import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "game")))
import rngs as rngs

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995 # cahnge this one 
batch_size = 64
replay_memory_size = 5000
max_episodes = 300
guess_range = 100
update_target_every = 10

replay_memory = []

# Custom Transformer Block
def TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64):
    inputs = layers.Input(shape=(1, embed_dim))
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = layers.Add()([inputs, attention_output])
    attention_output = layers.LayerNormalization()(attention_output)
    ffn_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    outputs = layers.Add()([attention_output, ffn_output])
    outputs = layers.LayerNormalization()(outputs)
    return keras.Model(inputs, outputs)

def create_model():
    inputs = keras.Input(shape=(1, 1))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Reshape((1, 32))(x)  # Reshape for Transformer input
    x = TransformerBlock()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)  # Predicts next number directly
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

model = create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.randint(1, guess_range)
    q_value = model.predict(np.array([[state]]).reshape(1, 1, 1), verbose=0)
    return int(np.round(q_value[0][0])) % guess_range

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
        target[i] = rewards[i] + discount_factor * target_q_values[i]
    model.fit(states, target, epochs=1, verbose=0)

def update_target_network():
    target_model.set_weights(model.get_weights())

def game_loop(game_func):
    global epsilon
    rewards = []
    for episode in range(max_episodes):
        state = random.randint(1, guess_range)
        total_reward = 0
        counter = 0
        while True:
            counter += 1
            action = choose_action(state)
            output = game_func(state)
            distance = abs(action - output)
            reward = -np.exp(distance / 20) if action != output else 100  # Always negative unless correct
            store_experience(state, action, reward, output)
            train_model()
            total_reward += reward
            if action == output:
                print(f"Episode {episode+1}: Solved in {counter} steps! The number was {output}. Reward: {total_reward}")
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

game_loop(rngs.not_random_addition)