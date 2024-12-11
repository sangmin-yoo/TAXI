DEPENDENCIES := src/main.o src/cost_function.o src/mylib.o src/problem.o src/ising_solver.o src/random_selector.o src/mid.o src/mid_grid.o src/distance_matrix.o
TARGET := ising_mac

all: $(TARGET)

$(TARGET): $(DEPENDENCIES)
	#g++-14 -O2 -arch arm64 $^ -o $@
	g++ -O2 $^ -o $@

%.o: %.cpp
	#g++-14 -std=c++14 -Wall -O2 -arch arm64 -c $^ -o $@
	g++ -std=c++14 -Wall -O2 -c $^ -o $@

clean:
	rm -f $(TARGET)
	rm -f src/*.o