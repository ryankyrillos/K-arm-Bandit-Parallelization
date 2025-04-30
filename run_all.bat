@echo off
REM Compile all versions
echo Compiling all versions...
make

REM Run sequential version
echo Running sequential version...
sequential_bandit > sequential_output.txt
findstr "Execution time:" sequential_output.txt

REM Run OpenMP version
echo Running OpenMP version...
set OMP_NUM_THREADS=4
openmp_bandit > openmp_output.txt
findstr "Execution time:" openmp_output.txt

REM Run MPI version
echo Running MPI version...
mpirun -np 4 mpi_bandit > mpi_output.txt
findstr "Execution time:" mpi_output.txt

REM Extract execution times
for /f "tokens=3" %%a in ('findstr "Execution time:" sequential_output.txt') do set seq_time=%%a
for /f "tokens=3" %%a in ('findstr "Execution time:" openmp_output.txt') do set omp_time=%%a
for /f "tokens=3" %%a in ('findstr "Execution time:" mpi_output.txt') do set mpi_time=%%a

REM Plot results
echo Plotting results...
python plot_results.py %seq_time% %omp_time% %mpi_time%

echo All done! Check the output files and plots.
