import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "game")))
import rngs as rngs

guess_range = 100
batch_size = 64
elite_frac = 0.2
max_iters = 500
num_rounds = 100  # Repeat the game 500 times

# ---- The environment ----
def number_guessing_game(target, guess):
    return -abs(target - guess)  # Negative reward if far, positive when correct

def rng_game(seed, target):
    # Use pseudo_random_lcg to generate the target number
    return rngs.pseudo_random_lcg(seed) % target + 1

# ---- CEM Algorithm ----
def cem_number_guess(target_number):
    # Initial probability distribution: uniform over [1, guess_range]
    mean = tf.Variable(guess_range / 2, dtype=tf.float32)
    std = tf.Variable(guess_range / 2, dtype=tf.float32)

    history = []
    
    for iteration in range(max_iters):
        round_correct = False
        round_history = []  # Store guesses and rewards for the round
        #print("test")
        while not round_correct:
            # Sample guesses
            samples = tf.random.normal((batch_size,), mean, std)
            samples = tf.clip_by_value(tf.round(samples), 1, guess_range)

            # Evaluate
            rewards = np.array([number_guessing_game(target_number, s.numpy()) for s in samples])

            # Keep elites
            n_elite = int(batch_size * elite_frac)
            elite_idxs = rewards.argsort()[-n_elite:]
            elite_samples = tf.gather(samples, elite_idxs)

            # Update mean & std towards elites
            new_mean = tf.reduce_mean(elite_samples)
            new_std = tf.math.reduce_std(elite_samples)

            mean.assign(new_mean)
            std.assign(new_std)

            # Track best guess and reward
            best_guess = elite_samples[-1].numpy()
            best_reward = rewards[elite_idxs[-1]]
            round_history.append((best_guess, best_reward))

            print(f"Iter {iteration+1}: Best Guess = {best_guess}, Reward = {best_reward}, Mean = {mean.numpy():.2f}, Std = {std.numpy():.2f}")

            # Check if we guessed correctly
            if abs(best_guess - target_number) == 0:
                print(f"✅ Guessed correctly in round!")
                round_correct = True  # Exit the while loop once correct guess is found

        history.append(round_history)  # Store round history for the whole game

    return history

# ---- Run Example ----
if __name__ == "__main__":
    all_histories = []
    
    for round_num in range(num_rounds):
        seed = random.randint(1, 100)  # Seed for the random number generator
        target = rng_game(seed, guess_range)
        print(f"Round {round_num+1}: Target number is {target}")
        round_history = cem_number_guess(target)
        all_histories.extend(round_history)

    # Plot progress
    # Collect all guesses across rounds for plotting
    all_guesses = [guess for round_hist in all_histories for guess, _ in round_hist]
    all_rewards = [reward for round_hist in all_histories for _, reward in round_hist]

    plt.plot(range(1, len(all_guesses) + 1), all_guesses, label="Guesses")
    plt.axhline(y=target, color='r', linestyle='--', label='Target')
    plt.xlabel('Iteration')
    plt.ylabel('Guess')
    plt.legend()
    plt.show()
