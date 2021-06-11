# Compile flags
CXXFLAGS = -fopenmp -Wall -Werror -I /usr/include/eigen3/ -Ofast -march=native -flto

# All...
all: svdistic

# Compile train programs
svdistic: main/svdistic.o models/svdpp/model.o models/svd/model.o models/base/base.o utils/pipes.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Clean all files
clean:
	rm *.o svdistic models/svdpp/*.o models/svd/*.o utils/*.o main/*.o models/base/*.o

# Main file
main/svdistic.o: main/svdistic.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build base
models/base/base.o: models/base/base.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build SVD++
models/svdpp/model.o: models/svdpp/model.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build SVD
models/svd/model.o: models/svd/model.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build utils
utils/pipes.o: utils/pipes.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

