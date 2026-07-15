
CXX = g++
CXXFLAGS = -std=c++17 -I engine -O3 -march=native -ffast-math -funroll-loops -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include

all:
	$(CXX) $(CXXFLAGS) engine/run.cpp -o a.out -L/opt/homebrew/opt/libomp/lib -lomp

clean:
	rm -f a.out
