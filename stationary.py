from numba import njit
import numpy as np
import matplotlib.pyplot as plt

@njit
def populate_arms(num_arms):
    return np.random.normal(0, 1, num_arms)

@njit
def generate_arm_return(mean):
    return np.random.normal(mean)

@njit
def select_greedy(sample_means):
    return np.argmax(sample_means)

@njit
def select_nongreedy(num_arms):
    return np.random.randint(0, num_arms)

@njit
def greedy_mask(iters, epsilon):
    return np.random.uniform(0, 1, iters) >= epsilon

@njit(fastmath=True)
def epsilon_greedy_iter(epsilon, num_arms, max_iter, initial):
    mask = greedy_mask(max_iter, epsilon)
    
    means = populate_arms(num_arms)

    sample_means = np.zeros(num_arms) + initial
    sample_count = np.zeros(num_arms)

    history_reward = np.zeros(max_iter)
    history_best = np.zeros(max_iter)

    best_arm = np.argmax(means)
    
    for iter in range(1, max_iter):
        pick = select_greedy(sample_means) if (mask[iter]) else select_nongreedy(num_arms)
        arm_return = generate_arm_return(means[pick])
        sample_count[pick] += 1
        sample_means[pick] += (arm_return - sample_means[pick])/sample_count[pick]
        history_reward[iter] = history_reward[iter-1] + (arm_return - history_reward[iter-1])/iter
        history_best[iter] = sample_count[best_arm]/iter
    return [history_reward, history_best]

@njit(fastmath=True, parallel=True)
def benchmark_epsilon_greedy(epsilon, num_arms, max_ensemble, max_iter, initial=0):
    history_reward_sum = np.zeros(max_iter)
    history_best_sum = np.zeros(max_iter)
    for _ in range(max_ensemble):
        [history_reward_iter, history_best_iter] = epsilon_greedy_iter(epsilon, num_arms, max_iter, initial)
        history_reward_sum += history_reward_iter
        history_best_sum += history_best_iter
    return [history_reward_sum/max_ensemble, history_best_sum/max_ensemble]

epsilons = [0, 0.1, 0.01]

history_rewards = []
history_bests = []

max_ensemble = 1000
max_iter = 2000
num_arms = 8
initial = 0

for epsilon in epsilons:
    [history_reward, history_best] = benchmark_epsilon_greedy(epsilon, num_arms, max_ensemble, max_iter, initial)
    history_rewards.append(history_reward)
    history_bests.append(history_best*100)

plt.figure()
for i in range(len(epsilons)):
    plt.plot(history_rewards[i], linewidth=0.5, label=epsilons[i])
plt.xlabel("timestep")
plt.ylabel("Average Reward")
plt.legend()

plt.savefig("./images/stationary_reward.svg")

plt.figure()
for i in range(len(epsilons)):
    plt.plot(history_bests[i], linewidth=0.5, label=epsilons[i])
plt.xlabel("timestep")
plt.ylabel("% best choice")
plt.legend()

plt.savefig("./images/stationary_best.svg")