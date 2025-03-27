import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "game")))
import rngs as rngs

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
guess_range = 100
max_episodes = 500

class A2CAgent:
    def __init__(self):
        self.actor, self.critic = self.create_model()
        self.actor_optimizer = keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate)

    def create_model(self):
        # Actor
        state_input = layers.Input(shape=(1,))
        x = layers.Dense(64, activation='relu')(state_input)
        x = layers.Dense(64, activation='relu')(x)
        action_probs = layers.Dense(guess_range, activation='softmax')(x)
        actor = keras.Model(inputs=state_input, outputs=action_probs)

        # Critic
        y = layers.Dense(64, activation='relu')(state_input)
        y = layers.Dense(64, activation='relu')(y)
        value = layers.Dense(1, activation='linear')(y)
        critic = keras.Model(inputs=state_input, outputs=value)

        return actor, critic

    def choose_action(self, state):
        state = np.array([state]).reshape(1, 1)
        probs = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(guess_range, p=probs)
        return action + 1, probs[action]

    def train(self, state, action, reward, next_state, done):
        state = np.array([state]).reshape(1, 1)
        next_state = np.array([next_state]).reshape(1, 1)
        action_idx = action - 1

        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(state, training=True)
            value = self.critic(state, training=True)
            next_value = self.critic(next_state, training=True)

            target = reward + (1 - int(done)) * gamma * next_value
            advantage = target - value

            action_prob = probs[0, action_idx]
            log_prob = tf.math.log(action_prob + 1e-10)
            actor_loss = -log_prob * advantage

            critic_loss = advantage ** 2

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

def game_loop(game_func):
    agent = A2CAgent()
    rewards = []

    for episode in range(max_episodes):
        state = random.randint(1, guess_range)
        total_reward = 0
        done = False
        counter = 0

        while not done and counter < 500:
            counter += 1
            action, prob = agent.choose_action(state)
            output = game_func(state)
            reward = 1000 if action == output else -abs(output - action)
            done = reward == 1000

            agent.train(state, action, reward, output, done)
            state = output
            total_reward += reward

        if counter > 99:
            total_reward -= 1000000
            print(f"Episode {episode+1}: FAILED - Took {counter} guesses")
        else:
            print(f"Episode {episode+1}: SUCCESS - Guessed in {counter} steps (Number: {output})")

        rewards.append(total_reward)

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

# Run it
game_loop(rngs.pseudo_random_lcg)
