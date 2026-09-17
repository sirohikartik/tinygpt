CXX = clang++
CXXFLAGS = -std=c++17 -I engine -O3 -march=native -ffast-math -funroll-loops -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -DACCELERATE_NEW_LAPACK
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate

ifeq ($(MPS), 1)
CXXFLAGS += -DUSE_MPS -DMPS
LDFLAGS += -framework Metal -framework MetalPerformanceShaders -framework Foundation
SRCS = engine/run.cpp engine/metal_kernels.mm
else
SRCS = engine/run.cpp
endif

all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o a.out $(LDFLAGS)

cpu:
	$(CXX) $(CXXFLAGS) engine/run.cpp -o a.out $(LDFLAGS)

mps:
	$(CXX) $(CXXFLAGS) -DUSE_MPS -DMPS engine/run.cpp engine/metal_kernels.mm -o a.out $(LDFLAGS) -framework Metal -framework MetalPerformanceShaders -framework Foundation

clean:
	rm -f a.out
