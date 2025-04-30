#!/bin/bash

# Compile all versions
echo "Compiling all versions..."
make

# Run sequential version
echo "Running sequential version..."
./sequential_bandit > sequential_output.txt
seq_time=$(grep "Execution time:" sequential_output.txt | awk '{print $3}')
echo "Sequential execution time: $seq_time seconds"

# Run OpenMP version
echo "Running OpenMP version..."
export OMP_NUM_THREADS=4  # Use 4 threads
./openmp_bandit > openmp_output.txt
omp_time=$(grep "Execution time:" openmp_output.txt | awk '{print $3}')
echo "OpenMP execution time: $omp_time seconds"

# Run MPI version
echo "Running MPI version..."
mpirun -np 4 ./mpi_bandit > mpi_output.txt
mpi_time=$(grep "Execution time:" mpi_output.txt | awk '{print $3}')
echo "MPI execution time: $mpi_time seconds"

# Plot results
echo "Plotting results..."
python plot_results.py $seq_time $omp_time $mpi_time

echo "All done! Check the output files and plots."
