override CXX := g++

override EIGEN_INCLUDE_DIR := $(CURDIR)/external/Eigen
override LOGICNN_PROJECT_ROOT := $(CURDIR)/external/LogicNN
override LOGICNN_INCLUDE_DIR := $(LOGICNN_PROJECT_ROOT)/include
override CXXFLAGS := -I$(LOGICNN_INCLUDE_DIR) -I$(EIGEN_INCLUDE_DIR) -Werror -Wextra -Wall -std=c++20 -O3 -mavx -mfma 
override LDFLAGS := -L./lib/LogicNN/release -llogicnn_backprop -llogicnn -lm
override SOURCES := $(wildcard src/*.cpp)
override HEADERS := $(wildcard include/*.hpp)
override OBJECTS := $(addprefix build/, $(SOURCES:src/%.cpp=%.o))

ADAMFILE := $(CURDIR)/result/adamfile
NNFILE := $(CURDIR)/result/nnfile
TRAINSET_DIR := $(CURDIR)/dataset/trainset
TESTSET_DIR := $(CURDIR)/dataset/testset
TRAINLOG := $(CURDIR)/results/trainlog.txt
TESTLOG := $(CURDIR)/results/testlog.txt
EPOCHS := 10
WORKERS := 4
ALPHA := 0.001
BATCH := 50

help:
	@echo "----------------------------------------------------------------------"
	@echo "make commands 								          	             "
	@echo "----------------------------------------------------------------------"
	@echo " 1. build   		> Builds training and test utilities for the model.  "
	@echo " 2. train   		> Trains the neural network.                         "
	@echo " 3. test    		> Evaluates the neural network on the test set.      "
	@echo " 4. run     		> Runs both training and test utilities.             "
	@echo " 5. restore 		> Replaces training state with the backup.           "
	@echo " 6. backup  		> Saves current training state.                      "
	@echo " 7. clean   		> Delete all build artifacts.                        "
	@echo "----------------------------------------------------------------------"

build: bin/trainer bin/tester

train:

test:

run: train test

restore:

backup:

clean:
	@rm -rf build bin lib
	@echo "Finished cleaning build artifacts!"

lib:
	@$(MAKE) -C $(LOGICNN_PROJECT_ROOT) BUILD_DIR=$(CURDIR)/build/LogicNN LIB_DIR=$(CURDIR)/lib/LogicNN release

bin/trainer: lib

bin/tester: lib

.PRECIOUS: build/%.o build/LogicNN/release/%.o
.PHONY: help build lib trainer tester train test run restore backup clean