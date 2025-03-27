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
learning_rate = 0.0005
clip_ratio = 0.2
gamma = 0.97
lambda_gae = 0.95
policy_update_epochs = 10
batch_size = 16
max_episodes = 500
guess_range = 100  # Range of guesses for game

# Actor-Critic Model
class PPOAgent:
    def __init__(self):
        self.actor, self.critic = self.create_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = []

    def create_model(self):
        state_input = layers.Input(shape=(1,))
        advantage = layers.Input(shape=(1,))
        old_log_probs = layers.Input(shape=(1,))
        
        common = layers.Dense(64, activation='relu')(state_input)
        common = layers.Dense(64, activation='relu')(common)
        
        action_probs = layers.Dense(guess_range, activation='softmax')(common)
        value = layers.Dense(1, activation='linear')(common)
        
        actor = keras.Model(inputs=state_input, outputs=action_probs)
        critic = keras.Model(inputs=state_input, outputs=value)
        
        return actor, critic
    
    def choose_action(self, state):
        state = np.array([state]).reshape(1, 1)
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(guess_range, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action + 1, log_prob
    
    def store_experience(self, state, action, reward, next_state, log_prob, done):
        self.memory.append((state, action, reward, next_state, log_prob, done))
    
    def train(self):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, log_probs, dones = zip(*batch)
        
        states = np.array(states).reshape(-1, 1)
        next_states = np.array(next_states).reshape(-1, 1)
        actions = np.array(actions) - 1
        old_log_probs = np.array(log_probs).reshape(-1, 1)
        
        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            adv = delta + gamma * lambda_gae * (1 - dones[t]) * adv
            advantages[t] = adv
            returns[t] = adv + values[t]
        
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        for _ in range(policy_update_epochs):
            with tf.GradientTape() as tape:
                probs = self.actor(states, training=True)
                action_probs = tf.gather(probs, actions[:, None], batch_dims=1)
                new_log_probs = tf.math.log(action_probs)
                ratio = tf.exp(new_log_probs - old_log_probs)
                
                clip_adv = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clip_adv))
                
                value_preds = self.critic(states, training=True)
                value_loss = tf.reduce_mean((returns - value_preds) ** 2)
                
                loss = policy_loss + 0.5 * value_loss
            
            grads = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))
        
        self.memory = []

# Training loop
def game_loop(game_func):
    agent = PPOAgent()
    rewards = []
    
    for episode in range(max_episodes):
        state = random.randint(1, guess_range)
        total_reward = 0
        done = False
        counter = 0
        states = []
        
        while not done and counter < 500:
            counter += 1
            action, log_prob = agent.choose_action(state)
            output = game_func(state)
            reward = 1 if action == output else -1
            done = reward == 1
            
            agent.store_experience(state, action, reward, output, log_prob, done)
            state = output
            states.append(state)
            total_reward += reward
            
        if counter > 99:
            total_reward -= 1000000
            print(f"Failed to guess in less that a hundred took {counter}")
        else:
            print(f"Episode {episode+1}: Correct Guess! The number was {output}. Took {counter} guesses")
        
        
        agent.train()
        
        rewards.append(total_reward)
    
    plt.plot(range(max_episodes), rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Run the game loop
game_loop(rngs.pseudo_random_lcg)