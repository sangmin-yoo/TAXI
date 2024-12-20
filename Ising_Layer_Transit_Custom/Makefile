# Compiler and flags
CXX = g++-14
CXXFLAGS = -std=c++14 -Wall -O2 -arch arm64

# Target executable name
TARGET = ising_mac_transit_custom

# Source files
SRCS = main.cpp

# Build target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)