
CXX = g++
CXXFLAGS = -std=c++17 -I engine -O3 -march=native -ffast-math -funroll-loops

all:
	$(CXX) $(CXXFLAGS) engine/run.cpp -o a.out

clean:
	rm -f a.out
