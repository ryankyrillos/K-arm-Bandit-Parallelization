# K-arm Bandit Problem Parallelization

This project implements the K-arm Bandit problem using sequential, OpenMP, and MPI approaches to compare their performance.

## Overview

The K-arm Bandit problem is a decision-making problem where you have a slot machine with K levers, each providing a different, unknown reward. The goal is to figure out which lever gives the highest reward by balancing exploration (trying different levers to learn their rewards) and exploitation (choosing the lever that has given the highest reward so far).

## Implementations

1. **Sequential Version**: A baseline implementation without parallelization.
2. **OpenMP Version**: Parallelizes the loops in action selection and action preference update functions using OpenMP.
3. **MPI Version**: Divides the total number of simulations equally among MPI processes.

## Requirements

- C++ compiler with C++11 support
- OpenMP
- MPI (e.g., MPICH or OpenMPI)
- Python 3 with matplotlib, numpy, and pandas (for plotting results)

## Building

To compile all versions, run:

```bash
make
```

This will create three executables:
- `sequential_bandit`
- `openmp_bandit`
- `mpi_bandit`

## Running

### Sequential Version

```bash
./sequential_bandit
```

### OpenMP Version

```bash
./openmp_bandit
```

You can set the number of OpenMP threads using the environment variable:

```bash
export OMP_NUM_THREADS=4  # Use 4 threads
./openmp_bandit
```

### MPI Version

```bash
mpirun -np 4 ./mpi_bandit  # Run with 4 MPI processes
```

## Plotting Results

After running all three versions, you can plot the results using:

```bash
python plot_results.py
```

To also plot execution time comparison, provide the execution times as arguments:

```bash
python plot_results.py <sequential_time> <openmp_time> <mpi_time>
```

For example:
```bash
python plot_results.py 10.5 3.2 2.8
```

## Parameters

When running any of the implementations, you'll be prompted to enter:
1. Number of arms (default: 10)
2. Length of each run (default: 1000)
3. Number of runs (default: 1000)
4. Exploration strategy:
   - 0: epsilon-greedy
   - 1: Boltzmann
   - 2: UCB
   - 3: gradient bandit

## Output

Each implementation generates a text file with the results:
- `sequential_results.txt`
- `openmp_results.txt`
- `mpi_results.txt`

The plotting script generates:
- `performance_comparison.png`: Comparing average reward and optimal action selection
- `execution_times.png`: Comparing execution times and speedup (if times are provided)
