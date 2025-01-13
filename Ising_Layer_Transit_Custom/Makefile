# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++14 -Wall

# Target executable name
TARGET = ising_mac_transit_custom_linux

# Source files
SRCS = main.cpp

# Build target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)