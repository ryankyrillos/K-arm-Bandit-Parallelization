import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def load_data(filename):
    try:
        data = pd.read_csv(filename, sep=' ')
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def plot_results():
    # Load data from all three implementations
    seq_data = load_data('sequential_results.txt')
    omp_data = load_data('openmp_results.txt')
    mpi_data = load_data('mpi_results.txt')
    
    if seq_data is None or omp_data is None or mpi_data is None:
        print("Error: Could not load one or more result files.")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot average reward over time
    ax1.plot(seq_data['step'], seq_data['avg_reward'], label='Sequential')
    ax1.plot(omp_data['step'], omp_data['avg_reward'], label='OpenMP')
    ax1.plot(mpi_data['step'], mpi_data['avg_reward'], label='MPI')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Average Reward over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Plot percentage of optimal action selections
    ax2.plot(seq_data['step'], seq_data['percentage_opt_action'] * 100, label='Sequential')
    ax2.plot(omp_data['step'], omp_data['percentage_opt_action'] * 100, label='MPI')
    ax2.plot(mpi_data['step'], mpi_data['percentage_opt_action'] * 100, label='OpenMP')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Percentage of Optimal Actions (%)')
    ax2.set_title('Percentage of Optimal Action Selections')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()
    
    print("Plot saved as 'performance_comparison.png'")

def plot_execution_times(seq_time, omp_time, mpi_time):
    # Create bar chart of execution times
    implementations = ['Sequential', 'OpenMP', 'MPI']
    times = [seq_time, omp_time, mpi_time]
    
    # Calculate speedup
    speedup_omp = seq_time / omp_time
    speedup_mpi = seq_time / mpi_time
    
    plt.figure(figsize=(10, 6))
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
    plt.figtext(0.15, 0.01, f"OpenMP Speedup: {speedup_omp:.2f}x", ha="center")
    plt.figtext(0.85, 0.01, f"MPI Speedup: {speedup_mpi:.2f}x", ha="center")
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('execution_times.png', dpi=300)
    plt.show()
    
    print("Execution time plot saved as 'execution_times.png'")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # If execution times are provided as command line arguments
        seq_time = float(sys.argv[1])
        omp_time = float(sys.argv[2])
        mpi_time = float(sys.argv[3])
        plot_execution_times(seq_time, omp_time, mpi_time)
    
    # Always plot the performance results
    plot_results()
