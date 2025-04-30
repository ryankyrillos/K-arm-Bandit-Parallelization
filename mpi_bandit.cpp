#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <mpi.h>

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

    // Take an action using UCB strategy
    int UCB(int t, double c) {
        int action = 0;

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

    // Take an action using Boltzmann exploration
    int Boltzmann_exploration(double T) {
        int action = 0;
        double max_val = 0.0;
        double denom = 0.0;
        std::vector<double> weights;

        for (int i = 0; i < N; i++) {
            q_temperature[i] = q[i] / T;
            if (q[i] > max_val) {
                max_val = q[i];
            }
        }

        for (int i = 0; i < N; i++) {
            denom += std::exp(q_temperature[i] - max_val);
        }

        for (int i = 0; i < N; i++) {
            weights.push_back(std::exp(q_temperature[i] - max_val) / denom);
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
        std::vector<double> pi;

        for (int i = 0; i < N; i++) {
            denom += std::exp(preferences[i]);
        }

        for (int i = 0; i < N; i++) {
            pii[i] = std::exp(preferences[i]) / denom;
            pi.push_back(std::exp(preferences[i]) / denom);
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

    // Update action preferences for gradient bandit
    void update_action_preferences(double r, int a) {
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

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parameters
    int N = 10;                // Number of arms
    int run_length = 1000;     // Length of each run
    int n_runs = 1000;         // Number of runs
    int exploration_strategy = 0; // 0: epsilon-greedy, 1: Boltzmann, 2: UCB, 3: gradient bandit
    double epsilon = 0.1;      // Exploration parameter
    double learning_rate = 0.1; // Learning rate
    double c = 2.0;            // UCB parameter
    double T = 0.1;            // Temperature for Boltzmann exploration

    // Only the root process gets input from the user
    if (rank == 0) {
        std::cout << "\n======\nWelcome to the K-arm Bandit simulator (MPI version)!\n";
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
            MPI_Abort(MPI_COMM_WORLD, -1);
            return -1;
        }

        std::cout << "Running with " << size << " MPI processes\n";
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&run_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_runs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exploration_strategy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&learning_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate the number of runs for each process
    int runs_per_process = n_runs / size;
    int remainder = n_runs % size;
    int start_run = rank * runs_per_process + (rank < remainder ? rank : remainder);
    int end_run = start_run + runs_per_process + (rank < remainder ? 1 : 0);
    int local_n_runs = end_run - start_run;

    if (rank == 0) {
        std::cout << "Total runs: " << n_runs << ", Runs per process: ~" << runs_per_process << "\n";
    }

    // Arrays to store local results
    std::vector<double> local_avg_rewards(run_length, 0.0);
    std::vector<double> local_percentage_best_action(run_length, 0.0);

    // Start timing
    double start_time = MPI_Wtime();

    // Run simulations assigned to this process
    for (int i = 0; i < local_n_runs; i++) {
        int global_i = start_run + i;
        
        if (rank == 0 && global_i % 100 == 0) {
            std::cout << "Run number " << global_i << "\n";
        }

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

        // Accumulate local results
        for (int j = 0; j < run_length; j++) {
            local_avg_rewards[j] += returns[j];
            if (opt_actions[j] == 1) {
                local_percentage_best_action[j] += 1.0;
            }
        }

        // Print true values for the first and last run (only on rank 0)
        if (rank == 0 && (global_i == 0 || global_i == n_runs - 1)) {
            b.print_true_values();
        }
    }

    // Reduce local results to the root process
    std::vector<double> global_avg_rewards(run_length, 0.0);
    std::vector<double> global_percentage_best_action(run_length, 0.0);

    MPI_Reduce(local_avg_rewards.data(), global_avg_rewards.data(), run_length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_percentage_best_action.data(), global_percentage_best_action.data(), run_length, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // End timing
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // Compute maximum elapsed time across all processes
    double global_elapsed;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Only the root process computes final results and writes to file
    if (rank == 0) {
        std::cout << "Execution time: " << global_elapsed << " seconds\n";

        // Normalize results
        for (int j = 0; j < run_length; j++) {
            global_avg_rewards[j] /= n_runs;
            global_percentage_best_action[j] /= n_runs;
        }

        // Save results to file
        std::ofstream outfile("mpi_results.txt");
        outfile << "step avg_reward percentage_opt_action\n";
        for (int j = 0; j < run_length; j++) {
            outfile << j << " " << global_avg_rewards[j] << " " << global_percentage_best_action[j] << "\n";
        }
        outfile.close();

        // Print summary
        std::cout << "Average reward at the end: " << global_avg_rewards[run_length - 1] << "\n";
        std::cout << "Percentage of optimal actions at the end: " << global_percentage_best_action[run_length - 1] * 100 << "%\n";
        std::cout << "Results saved to mpi_results.txt\n";
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
