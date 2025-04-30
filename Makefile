CXX = mpicxx
CXXFLAGS = -std=c++11 -Wall -O3
OMPFLAGS = -fopenmp

all: sequential openmp mpi

sequential: sequential_bandit.cpp
	$(CXX) $(CXXFLAGS) -o sequential_bandit sequential_bandit.cpp

openmp: openmp_bandit.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o openmp_bandit openmp_bandit.cpp

mpi: mpi_bandit.cpp
	$(CXX) $(CXXFLAGS) -o mpi_bandit mpi_bandit.cpp

clean:
	rm -f sequential_bandit openmp_bandit mpi_bandit *.o *_results.txt

.PHONY: all sequential openmp mpi clean
