import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import os
import threading

# Bandit class for the K-arm bandit problem
class Bandit:
    def __init__(self, n_arms=10, epsilon=0.1, learning_rate=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q = np.zeros(n_arms)  # Estimated values
        self.true_values = np.random.normal(0, 1, n_arms)  # True values
        self.n_actions = np.zeros(n_arms, dtype=int)  # Number of times each arm was selected
        self.best_action = 0  # Whether the last action was the best action

    def take_action(self):
        """Take an action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:  # Random action
            action = np.random.randint(0, self.n_arms)
        else:  # Greedy action
            action = np.argmax(self.q)

        # Check if this is the best action
        if action == np.argmax(self.true_values):
            self.best_action = 1
        else:
            self.best_action = 0

        self.n_actions[action] += 1
        return action

    def ucb(self, t, c=2.0):
        """Take an action using UCB strategy"""
        ucb_values = np.zeros(self.n_arms)
        for j in range(self.n_arms):
            if self.n_actions[j] != 0:
                ucb_values[j] = self.q[j] + c * np.sqrt(np.log(t) / self.n_actions[j])
            else:
                ucb_values[j] = 10000  # Very high value to ensure exploration

        action = np.argmax(ucb_values)

        # Check if this is the best action
        if action == np.argmax(self.true_values):
            self.best_action = 1
        else:
            self.best_action = 0

        self.n_actions[action] += 1
        return action

    def get_reward(self, action):
        """Get the reward for an action"""
        return np.random.normal(self.true_values[action], 1.0)

    def update_q(self, reward, action):
        """Update the estimated value of an arm"""
        self.q[action] += self.learning_rate * (reward - self.q[action])

    def get_best_action(self):
        """Get whether the last action was the best action"""
        return self.best_action

# Function to run a single experiment
def run_experiment(run_length=1000, n_arms=10, epsilon=0.1, learning_rate=0.1, exploration_strategy=0, c=2.0):
    """Run a single experiment and return the rewards and optimal actions"""
    bandit = Bandit(n_arms, epsilon, learning_rate)
    rewards = np.zeros(run_length)
    opt_actions = np.zeros(run_length)

    for j in range(run_length):
        if exploration_strategy == 0:  # epsilon-greedy
            action = bandit.take_action()
        elif exploration_strategy == 2:  # UCB
            action = bandit.ucb(j, c)
        else:
            action = bandit.take_action()  # Default to epsilon-greedy

        opt_actions[j] = bandit.get_best_action()
        reward = bandit.get_reward(action)
        rewards[j] = reward
        bandit.update_q(reward, action)

    return rewards, opt_actions

# Sequential implementation
def sequential_simulation(n_runs=1000, run_length=1000, n_arms=10, epsilon=0.1, learning_rate=0.1, exploration_strategy=0, c=2.0):
    """Run the simulation sequentially"""
    start_time = time.time()

    all_rewards = np.zeros((n_runs, run_length))
    all_opt_actions = np.zeros((n_runs, run_length))

    for i in range(n_runs):
        if i % 100 == 0:
            print(f"Run {i}/{n_runs}")
        rewards, opt_actions = run_experiment(run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)
        all_rewards[i] = rewards
        all_opt_actions[i] = opt_actions

    # Calculate average rewards and percentage of optimal actions
    avg_rewards = np.mean(all_rewards, axis=0)
    percentage_best_action = np.mean(all_opt_actions, axis=0)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Sequential execution time: {execution_time:.2f} seconds")

    # Save results to file
    np.savetxt('sequential_results.txt', np.column_stack((np.arange(run_length), avg_rewards, percentage_best_action)),
               header='step avg_reward percentage_opt_action', fmt='%d %.6f %.6f')

    return avg_rewards, percentage_best_action, execution_time

# Parallel implementation using multiprocessing (OpenMP-like)
def run_experiment_wrapper(i, run_length=1000, n_arms=10, epsilon=0.1, learning_rate=0.1, exploration_strategy=0, c=2.0):
    """Wrapper function for parallel processing"""
    if i % 100 == 0:
        print(f"Run {i}/{1000}")
    return run_experiment(run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)

def openmp_simulation(n_runs=1000, run_length=1000, n_arms=10, epsilon=0.1, learning_rate=0.1, exploration_strategy=0, c=2.0):
    """Run the simulation in parallel using multiprocessing (similar to OpenMP)"""
    start_time = time.time()

    # Get the number of CPU cores
    num_cores = mp.cpu_count()
    print(f"Running with {num_cores} CPU cores (OpenMP-like)")

    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Run the experiments in parallel
        args = [(i, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c) for i in range(n_runs)]
        results = pool.starmap(run_experiment_wrapper, args)

    # Extract rewards and optimal actions
    all_rewards = np.array([r[0] for r in results])
    all_opt_actions = np.array([r[1] for r in results])

    # Calculate average rewards and percentage of optimal actions
    avg_rewards = np.mean(all_rewards, axis=0)
    percentage_best_action = np.mean(all_opt_actions, axis=0)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"OpenMP-like execution time: {execution_time:.2f} seconds")

    # Save results to file
    np.savetxt('openmp_results.txt', np.column_stack((np.arange(run_length), avg_rewards, percentage_best_action)),
               header='step avg_reward percentage_opt_action', fmt='%d %.6f %.6f')

    return avg_rewards, percentage_best_action, execution_time

# Parallel implementation using threads (MPI-like)
def thread_worker(thread_id, n_runs_per_thread, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c, results):
    """Worker function for thread-based parallelism (similar to MPI)"""
    local_rewards = np.zeros((n_runs_per_thread, run_length))
    local_opt_actions = np.zeros((n_runs_per_thread, run_length))

    start_idx = thread_id * n_runs_per_thread
    end_idx = start_idx + n_runs_per_thread

    for i in range(start_idx, end_idx):
        if i % 100 == 0:
            print(f"Run {i}/{n_runs_per_thread * 4}")

        rewards, opt_actions = run_experiment(run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)
        local_rewards[i - start_idx] = rewards
        local_opt_actions[i - start_idx] = opt_actions

    # Store results in the shared dictionary
    results[thread_id] = (local_rewards, local_opt_actions)

def mpi_simulation(n_runs=1000, run_length=1000, n_arms=10, epsilon=0.1, learning_rate=0.1, exploration_strategy=0, c=2.0):
    """Run the simulation using threads to simulate MPI processes"""
    start_time = time.time()

    # Simulate 4 MPI processes
    n_processes = 4
    print(f"Running with {n_processes} simulated MPI processes")

    # Calculate runs per process
    n_runs_per_process = n_runs // n_processes

    # Create threads
    threads = []
    results = {}

    for i in range(n_processes):
        thread = threading.Thread(
            target=thread_worker,
            args=(i, n_runs_per_process, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c, results)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Combine results from all threads
    all_rewards = np.vstack([results[i][0] for i in range(n_processes)])
    all_opt_actions = np.vstack([results[i][1] for i in range(n_processes)])

    # Calculate average rewards and percentage of optimal actions
    avg_rewards = np.mean(all_rewards, axis=0)
    percentage_best_action = np.mean(all_opt_actions, axis=0)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"MPI-like execution time: {execution_time:.2f} seconds")

    # Save results to file
    np.savetxt('mpi_results.txt', np.column_stack((np.arange(run_length), avg_rewards, percentage_best_action)),
               header='step avg_reward percentage_opt_action', fmt='%d %.6f %.6f')

    return avg_rewards, percentage_best_action, execution_time

# Plot results
def plot_results(seq_rewards, seq_opt_actions, openmp_rewards, openmp_opt_actions, mpi_rewards, mpi_opt_actions, seq_time, openmp_time, mpi_time):
    """Plot the results and compare performance"""
    # Create figure with two subplots
    plt.figure(figsize=(15, 6))

    # Plot average reward over time
    plt.subplot(1, 2, 1)
    plt.plot(seq_rewards, label='Sequential')
    plt.plot(openmp_rewards, label='OpenMP')
    plt.plot(mpi_rewards, label='MPI')
    plt.xlabel('Time Step')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Time')
    plt.legend()
    plt.grid(True)

    # Plot percentage of optimal action selections
    plt.subplot(1, 2, 2)
    plt.plot(seq_opt_actions * 100, label='Sequential')
    plt.plot(openmp_opt_actions * 100, label='OpenMP')
    plt.plot(mpi_opt_actions * 100, label='MPI')
    plt.xlabel('Time Step')
    plt.ylabel('Percentage of Optimal Actions (%)')
    plt.title('Percentage of Optimal Action Selections')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)

    # Create bar chart of execution times
    plt.figure(figsize=(10, 6))
    implementations = ['Sequential', 'OpenMP', 'MPI']
    times = [seq_time, openmp_time, mpi_time]

    # Calculate speedups
    speedup_openmp = seq_time / openmp_time
    speedup_mpi = seq_time / mpi_time

    bars = plt.bar(implementations, times, color=['blue', 'green', 'red'])

    # Add execution time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom')

    plt.xlabel('Implementation')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add speedup text
    plt.figtext(0.25, 0.01, f"OpenMP Speedup: {speedup_openmp:.2f}x", ha="center")
    plt.figtext(0.75, 0.01, f"MPI Speedup: {speedup_mpi:.2f}x", ha="center")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('execution_times.png', dpi=300)

    print("Plots saved as 'performance_comparison.png' and 'execution_times.png'")

if __name__ == "__main__":
    # Parameters
    n_arms = 10
    run_length = 1000
    n_runs = 1000
    exploration_strategy = 0  # 0: epsilon-greedy, 2: UCB
    epsilon = 0.1
    learning_rate = 0.1
    c = 2.0  # UCB parameter

    print("\n======\nWelcome to the K-arm Bandit simulator!\n")
    print(f"Running with parameters:")
    print(f"Number of arms: {n_arms}")
    print(f"Length of each run: {run_length}")
    print(f"Number of runs: {n_runs}")
    print(f"Exploration strategy: {exploration_strategy} (0: epsilon-greedy, 2: UCB)")
    print(f"Epsilon: {epsilon}")
    print(f"Learning rate: {learning_rate}")
    print(f"UCB parameter c: {c}")

    # Run sequential simulation
    print("\nRunning sequential simulation...")
    seq_rewards, seq_opt_actions, seq_time = sequential_simulation(
        n_runs, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)

    # Run OpenMP-like simulation
    print("\nRunning OpenMP-like simulation...")
    openmp_rewards, openmp_opt_actions, openmp_time = openmp_simulation(
        n_runs, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)

    # Run MPI-like simulation
    print("\nRunning MPI-like simulation...")
    mpi_rewards, mpi_opt_actions, mpi_time = mpi_simulation(
        n_runs, run_length, n_arms, epsilon, learning_rate, exploration_strategy, c)

    # Plot results
    print("\nPlotting results...")
    plot_results(
        seq_rewards, seq_opt_actions,
        openmp_rewards, openmp_opt_actions,
        mpi_rewards, mpi_opt_actions,
        seq_time, openmp_time, mpi_time
    )

    print("\nAll done! Check the output files and plots.")
