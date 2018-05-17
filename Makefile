# The flags
CXXFLAGS = -Wall -Werror -I /usr/local/include/eigen3/ -O3
# CXXFLAGS = -Wall -Werror -I /usr/local/include/eigen3/ -g

# All
all: train_svdpp train_svd test_svd test_svdpp test_utils

# Compile train programs
train_svdpp: svdpp.o svdpp/model.o utils/pipes.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
train_svd: svd.o svd/model.o utils/pipes.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile test programs
test_svdpp: svdpp/tests.o svdpp/model.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 
test_svd: svd/tests.o svd/model.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 
test_utils: utils/tests.o utils/pipes.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) 

# Clean all files
clean:
	rm *.o test_svdpp test_svd train_svdpp train_svd \
     test_utils svdpp/*.o svd/*.o utils/*.o

# Build SVD++
svdpp.o: svdpp.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
svdpp/model.o: svdpp/model.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build SVD
svd.o: svd.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
svd/model.o: svd/model.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build utils
utils/pipes.o: utils/pipes.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Build all tests
svdpp/tests.o: svdpp/tests.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
svd/tests.o: svd/tests.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
utils/tests.o: utils/tests.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

