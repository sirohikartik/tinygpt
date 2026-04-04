CXX = g++
CXXFLAGS = -std=c++17 -I engine 

all:
	$(CXX) $(CXXFLAGS) engine/run.cpp -o a.out

clean:
	rm -f a.out


