
CXX = g++
CXXFLAGS = -std=c++17 -I engine -O3 -march=native -ffast-math -funroll-loops -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -DACCELERATE_NEW_LAPACK

all:
	$(CXX) $(CXXFLAGS) engine/run.cpp -o a.out -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate

clean:
	rm -f a.out
