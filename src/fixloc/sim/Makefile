CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17
ifeq ($(shell uname),Linux)
  CXXFLAGS += -fopenmp
endif

sweepsim: sweepsim.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f sweepsim
