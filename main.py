import numpy as np
import matplotlib.pyplot as plt

def populate_means(num_samples):
    return np.random.normal(0, 1, num_samples)

def get_arm_return(selected_sample, means):
    return np.random.normal(means[selected_sample], 1, 1)

def sample_uniform(idx_range, exclude):
    pick_range = np.arange(idx_range)
    sample_range = np.delete(pick_range, exclude)
    return np.random.choice(sample_range, 1)

def is_greedy(epsilon):
    return np.random.uniform() >= epsilon

def choose_epsilon_greedy(pick_sum, pick_count, epsilon):
    means = pick_sum/pick_count
    maxarg = np.argmax(means)
    if is_greedy(epsilon):
        return maxarg
    else:
        return sample_uniform(len(pick_sum), maxarg)

def benchmark_policy(epsilon):
    max_ensemble = 100
    max_iter = 1000
    arm_length = 10

    pick_sum = np.zeros(arm_length)
    pick_count = np.zeros(arm_length)

    # history for data anaylsis
    reward_history = np.zeros(max_iter)
    best_pick_history = np.zeros(max_iter)

    for ensemble in range(max_ensemble):
        means = populate_means(arm_length)
        best_arm = np.argmax(means)
        pick_sum.fill(0)
        pick_count.fill(0)
        for iter in range(max_iter):
            choice = choose_epsilon_greedy(pick_sum, pick_count, epsilon)
            return_value = get_arm_return(choice, means)
            pick_count[choice] += 1
            pick_sum[choice] += return_value

            reward_history[iter] += np.sum(pick_sum)/(iter+1)
            best_pick_history[iter] += pick_count[best_arm]/(iter+1)
    
    return [reward_history/max_ensemble, best_pick_history/max_ensemble]


def main():
    [reward_history0, best_pick_history0] = benchmark_policy(0)
    [reward_history1, best_pick_history1] = benchmark_policy(0.01)
    [reward_history2, best_pick_history2] = benchmark_policy(0.001)

    plt.figure()
    plt.plot(reward_history0)
    plt.plot(reward_history1, linestyle="--")
    plt.plot(reward_history2, linestyle="-.")
    plt.title("average reward")
    plt.xlabel("time step")
    plt.ylabel("average reward")

    plt.figure()
    plt.plot(best_pick_history0*100)
    plt.plot(best_pick_history1*100, linestyle="--")
    plt.plot(best_pick_history2*100, linestyle="-.")
    plt.title("best choice ratio")
    plt.xlabel("time step")
    plt.ylabel("%% best choice ratio")
    plt.show()


if __name__ == "__main__":
    main()
