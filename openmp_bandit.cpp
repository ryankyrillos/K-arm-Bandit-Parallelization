#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>

// Bandit class for the K-arm bandit problem
class Bandit {
private:
    int N;                  // Number of arms
    double epsilon;         // Exploration parameter
    double learning_rate;   // Learning rate
    std::vector<double> q;  // Estimated values
    std::vector<double> preferences; // Action preferences for gradient bandit
    std::vector<double> true_values; // True values of each arm
    std::vector<int> nt;    // Number of times each arm was selected
    std::vector<double> UCBvalues; // UCB values
    std::vector<double> q_temperature; // Temperature values for Boltzmann
    std::vector<double> pii; // Probability of selecting each arm
    double avg_reward;      // Average reward
    int best_action;        // Whether the last action was the best action

    // Random number generators
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    // Constructor
    Bandit(int n, double e, double l) : 
        N(n), 
        epsilon(e), 
        learning_rate(l), 
        q(n, 0.0), 
        preferences(n, 0.0), 
        true_values(n), 
        nt(n, 0), 
        UCBvalues(n, 0.0), 
        q_temperature(n, 0.0), 
        pii(n, 0.0), 
        avg_reward(0.0), 
        best_action(0),
        gen(rd()),
        dis(0.0, 1.0) {
        
        // Initialize true values with random normal distribution
        std::normal_distribution<> normal_dist(0.0, 1.0);
        for (int i = 0; i < N; i++) {
            true_values[i] = normal_dist(gen);
        }
    }

    // Take an action using epsilon-greedy strategy
    int take_action() {
        int action = 0;
        double rand_num = dis(gen);

        if (rand_num < epsilon) {  // Random action
            std::uniform_int_distribution<> int_dist(0, N - 1);
            action = int_dist(gen);
        } else {  // Greedy action
            action = std::distance(q.begin(), std::max_element(q.begin(), q.end()));
        }

        // Check if this is the best action
        if (action == std::distance(true_values.begin(), std::max_element(true_values.begin(), true_values.end()))) {
            best_action = 1;
        } else {
            best_action = 0;
        }

        nt[action]++;
        return action;
    }

    // Take an action using UCB strategy - Parallelized with OpenMP
    int UCB(int t, double c) {
        int action = 0;

        // Parallelize the loop that calculates UCB values
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            if (nt[j] != 0) {
                UCBvalues[j] = q[j] + c * std::sqrt(std::log(static_cast<float>(t)) / nt[j]);
            } else {
                UCBvalues[j] = 10000;  // Very high value to ensure exploration
            }
        }
        action = std::distance(UCBvalues.begin(), std::max_element(UCBvalues.begin(), UCBvalues.end()));

        // Check if this is the best action
        if (action == std::distance(true_values.begin(), std::max_element(true_values.begin(), true_values.end()))) {
            best_action = 1;
        } else {
            best_action = 0;
        }

        nt[action]++;
        return action;
    }

    // Take an action using Boltzmann exploration - Parallelized with OpenMP
    int Boltzmann_exploration(double T) {
        int action = 0;
        double max_val = 0.0;
        double denom = 0.0;
        std::vector<double> weights(N);

        // Parallelize the loop that calculates temperature values
        #pragma omp parallel
        {
            // First find the maximum value
            #pragma omp for reduction(max:max_val)
            for (int i = 0; i < N; i++) {
                q_temperature[i] = q[i] / T;
                if (q[i] > max_val) {
                    max_val = q[i];
                }
            }

            // Calculate the denominator
            #pragma omp for reduction(+:denom)
            for (int i = 0; i < N; i++) {
                denom += std::exp(q_temperature[i] - max_val);
            }

            // Calculate the weights
            #pragma omp for
            for (int i = 0; i < N; i++) {
                weights[i] = std::exp(q_temperature[i] - max_val) / denom;
            }
        }

        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        action = distribution(gen);

        // Check if this is the best action
        if (action == std::distance(true_values.begin(), std::max_element(true_values.begin(), true_values.end()))) {
            best_action = 1;
        } else {
            best_action = 0;
        }

        return action;
    }

    // Take an action using gradient bandit
    int gradientBanditAction() {
        double denom = 0.0;
        int action;
        std::vector<double> pi(N);

        // Calculate denominator
        #pragma omp parallel for reduction(+:denom)
        for (int i = 0; i < N; i++) {
            denom += std::exp(preferences[i]);
        }

        // Calculate probabilities
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            pii[i] = std::exp(preferences[i]) / denom;
            pi[i] = pii[i];
        }

        std::discrete_distribution<int> distribution(pi.begin(), pi.end());
        action = distribution(gen);

        // Check if this is the best action
        if (action == std::distance(true_values.begin(), std::max_element(true_values.begin(), true_values.end()))) {
            best_action = 1;
        } else {
            best_action = 0;
        }

        return action;
    }

    // Update the estimated value of an arm
    void update_q(double r, int a) {
        q[a] += learning_rate * (r - q[a]);
    }

    // Update the average reward
    void update_avg_reward(int n, double r) {
        if (n == 0) {
            avg_reward += 1 * (r - avg_reward);
        } else {
            avg_reward += 1.0 / n * (r - avg_reward);
        }
    }

    // Update action preferences for gradient bandit - Parallelized with OpenMP
    void update_action_preferences(double r, int a) {
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            if (i == a) {
                preferences[i] += learning_rate * (r - avg_reward) * (1 - pii[i]);
            } else {
                preferences[i] -= learning_rate * (r - avg_reward) * pii[i];
            }
        }
    }

    // Get the reward for an action
    double get_reward(int action) {
        // Normal distribution with mean = true value and std = 1.0
        std::normal_distribution<> dist(true_values[action], 1.0);
        return dist(gen);
    }

    // Get whether the last action was the best action
    int get_best_action() {
        return best_action;
    }

    // Print the true values
    void print_true_values() {
        std::cout << "\nTrue values:\n";
        for (int i = 0; i < N; i++) {
            std::cout << "Arm " << i << ": " << true_values[i] << "\n";
        }
        std::cout << "\n";
    }
};

// Experiment class to run simulations
class Experiment {
private:
    double epsilon;
    double learning_rate;
    int run_length;
    std::vector<double> returns;
    std::vector<int> opt_actions;

public:
    // Constructor
    Experiment(double e, double l, int rl) : 
        epsilon(e), 
        learning_rate(l), 
        run_length(rl), 
        returns(rl, 0.0), 
        opt_actions(rl, 0) {}

    // Run a single experiment with epsilon-greedy
    void single_run(Bandit &b) {
        int a = 0;
        double r = 0.0;

        for (int j = 0; j < run_length; j++) {
            a = b.take_action();
            opt_actions[j] = b.get_best_action();
            r = b.get_reward(a);
            returns[j] = r;
            b.update_q(r, a);
        }
    }

    // Run a single experiment with UCB
    void single_run_UCB(Bandit &b, double c) {
        int a = 0;
        double r = 0.0;

        for (int j = 0; j < run_length; j++) {
            a = b.UCB(j, c);
            opt_actions[j] = b.get_best_action();
            r = b.get_reward(a);
            returns[j] = r;
            b.update_q(r, a);
        }
    }

    // Run a single experiment with Boltzmann exploration
    void single_run_Boltzmann(Bandit &b, double T) {
        int a = 0;
        double r = 0.0;

        for (int j = 0; j < run_length; j++) {
            a = b.Boltzmann_exploration(T);
            opt_actions[j] = b.get_best_action();
            r = b.get_reward(a);
            returns[j] = r;
            b.update_q(r, a);
        }
    }

    // Run a single experiment with gradient bandit
    void single_run_gradient(Bandit &b) {
        int a = 0;
        double r = 0.0;

        for (int j = 0; j < run_length; j++) {
            a = b.gradientBanditAction();
            opt_actions[j] = b.get_best_action();
            r = b.get_reward(a);
            returns[j] = r;
            b.update_avg_reward(j, r);
            b.update_action_preferences(r, a);
        }
    }

    // Get the returns
    const std::vector<double>& get_returns() const {
        return returns;
    }

    // Get the optimal actions
    const std::vector<int>& get_opt_actions() const {
        return opt_actions;
    }
};

int main() {
    // Parameters
    int N = 10;                // Number of arms
    int run_length = 1000;     // Length of each run
    int n_runs = 1000;         // Number of runs
    int exploration_strategy = 0; // 0: epsilon-greedy, 1: Boltzmann, 2: UCB, 3: gradient bandit
    double epsilon = 0.1;      // Exploration parameter
    double learning_rate = 0.1; // Learning rate
    double c = 2.0;            // UCB parameter
    double T = 0.1;            // Temperature for Boltzmann exploration

    // Ask user for parameters
    std::cout << "\n======\nWelcome to the K-arm Bandit simulator (OpenMP version)!\n";
    std::cout << "Enter the number of arms (default 10): ";
    std::cin >> N;
    std::cout << "Enter the length of each run (default 1000): ";
    std::cin >> run_length;
    std::cout << "Enter the number of runs (default 1000): ";
    std::cin >> n_runs;
    std::cout << "Enter the exploration strategy (0: epsilon-greedy, 1: Boltzmann, 2: UCB, 3: gradient bandit): ";
    std::cin >> exploration_strategy;

    if (exploration_strategy > 3 || exploration_strategy < 0) {
        std::cout << "*****\nUnexpected number for the exploration strategy\n*****\n";
        return -1;
    }

    // Get the number of threads
    int num_threads = omp_get_max_threads();
    std::cout << "Running with " << num_threads << " OpenMP threads\n";

    // Arrays to store results
    std::vector<double> all_returns(n_runs * run_length);
    std::vector<int> all_opt_actions(n_runs * run_length);
    std::vector<double> avg_rewards(run_length, 0.0);
    std::vector<double> percentage_best_action(run_length, 0.0);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run simulations in parallel using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_runs; i++) {
        // Thread-local variables for results
        std::vector<double> local_returns(run_length);
        std::vector<int> local_opt_actions(run_length);

        // Print progress (only from master thread)
        #pragma omp critical
        {
            if (i % 100 == 0) {
                std::cout << "Run number " << i << "\n";
            }
        }

        // Create a thread-local bandit and experiment
        Bandit b(N, epsilon, learning_rate);
        Experiment e(epsilon, learning_rate, run_length);

        // Run the experiment with the selected strategy
        switch (exploration_strategy) {
            case 0:
                e.single_run(b);
                break;
            case 1:
                e.single_run_Boltzmann(b, T);
                break;
            case 2:
                e.single_run_UCB(b, c);
                break;
            case 3:
                e.single_run_gradient(b);
                break;
        }

        // Get results
        const std::vector<double>& returns = e.get_returns();
        const std::vector<int>& opt_actions = e.get_opt_actions();

        // Store results in the global arrays
        #pragma omp critical
        {
            for (int j = 0; j < run_length; j++) {
                all_returns[i * run_length + j] = returns[j];
                all_opt_actions[i * run_length + j] = opt_actions[j];
            }

            // Print true values for the first and last run
            if (i == 0 || i == n_runs - 1) {
                b.print_true_values();
            }
        }
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    // Compute average rewards and percentage of optimal actions
    for (int j = 0; j < run_length; j++) {
        for (int i = 0; i < n_runs; i++) {
            avg_rewards[j] += all_returns[i * run_length + j];
            if (all_opt_actions[i * run_length + j] == 1) {
                percentage_best_action[j] += 1.0 / n_runs;
            }
        }
        avg_rewards[j] /= n_runs;
    }

    // Save results to file
    std::ofstream outfile("openmp_results.txt");
    outfile << "step avg_reward percentage_opt_action\n";
    for (int j = 0; j < run_length; j++) {
        outfile << j << " " << avg_rewards[j] << " " << percentage_best_action[j] << "\n";
    }
    outfile.close();

    // Print summary
    std::cout << "Average reward at the end: " << avg_rewards[run_length - 1] << "\n";
    std::cout << "Percentage of optimal actions at the end: " << percentage_best_action[run_length - 1] * 100 << "%\n";
    std::cout << "Results saved to openmp_results.txt\n";

    return 0;
}
