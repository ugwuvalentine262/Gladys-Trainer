
# -------------------------- variables --------------------------

EPOCHS := 1
SESSIONS := 1
WORKERS := 4
ALPHA := 0.001
BATCH := 50
BREAK := 10
LOG ?= FALSE
EMBEDDING ?= 8 # specified at compile-time

override CXX := g++
override BACKUP_DIR := $(CURDIR)/backup
override RESULT_DIR := $(CURDIR)/result
override DATASET_DIR := $(CURDIR)/dataset
override ADAMFILE := adam
override NNFILE := nn
override LOGFILE := sessions.txt
override TRAINLOG := trainlog.txt
override TESTLOG := testlog.txt
override TRAINSET_DIR := $(DATASET_DIR)/trainset
override TESTSET_DIR := $(DATASET_DIR)/testset
override LOGICNN_PROJECT_ROOT := $(CURDIR)/external/LogicNN
override LOGICNN_INCLUDE_DIR := $(LOGICNN_PROJECT_ROOT)/include
override EIGEN_INCLUDE_DIR := $(CURDIR)/external/Eigen
override CXXFLAGS := -I$(LOGICNN_INCLUDE_DIR) -I$(EIGEN_INCLUDE_DIR) -Werror -Wextra -Wall -std=c++23 -O3 -march=native
override LDFLAGS := -L$(CURDIR)/lib/LogicNN/release -llogicnn_backprop -llogicnn -lm
override TRAINER_SRC := $(filter-out src/tester.cpp, $(wildcard src/*.cpp))
override TRAINER_HDR := $(filter-out src/tester.hpp, $(wildcard src/*.hpp))
override TRAINER_OBJ := $(addprefix build/trainer/, $(TRAINER_SRC:src/%.cpp=%.o))
override TESTER_SRC := $(filter-out src/trainer.cpp src/adam.cpp, $(wildcard src/*.cpp))
override TESTER_HDR := $(filter-out src/trainer.hpp src/adam.hpp, $(wildcard src/*.hpp))
override TESTER_OBJ := $(addprefix build/tester/, $(TESTER_SRC:src/%.cpp=%.o))

ifeq ($(LOG), TRUE)
override OUTPUT := >> $(RESULT_DIR)/$(LOGFILE)
else ifeq ($(LOG), ON)
override OUTPUT := >> $(RESULT_DIR)/$(LOGFILE)
else ifeq ($(LOG), 1)
override OUTPUT := >> $(RESULT_DIR)/$(LOGFILE)
endif

# -------------------------- macros --------------------------

define copy
	@rm -rf $(2)
	@mkdir -p $(2)
	@cp $(1)/$(NNFILE)   $(2)/$(NNFILE)   2>/dev/null || true
	@cp $(1)/$(ADAMFILE) $(2)/$(ADAMFILE) 2>/dev/null || true
	@cp $(1)/$(TRAINLOG) $(2)/$(TRAINLOG) 2>/dev/null || true
	@cp $(1)/$(TESTLOG)  $(2)/$(TESTLOG)  2>/dev/null || true
	@cp $(1)/$(LOGFILE)  $(2)/$(LOGFILE)  2>/dev/null || true
endef

# -------------------------- rules --------------------------

help:
	@echo "-----------------------------------------------------------------------"
	@echo "make commands                                                          "
	@echo "-----------------------------------------------------------------------"
	@echo "* build        > Builds training and test utilities for the model.     "
	@echo "* rebuild      > Rebuilds everything from scratch.                     "
	@echo "* init         > Initializes the parameters of the model.              "
	@echo "* run          > Runs $(SESSIONS) session(s) of training.              "
	@echo "* epoch        > Trains the model for a single epoch.                  "
	@echo "* eval         > Evaluates the neural network on the test set.         "
	@echo "* backup       > Saves current training state.                         "
	@echo "* restore      > Replaces training state with the backup.              "
	@echo "* refresh      > Deletes file containing optimizer state.              "
	@echo "* reset        > Resets training states to square one.                 "
	@echo "* clean        > Delete all build artifacts.                           "
	@echo "-----------------------------------------------------------------------"

build: bin/trainer bin/tester
	@echo "Finished building training and test utilities!"

rebuild:
	@rm -rf build bin lib
	@$(MAKE) build >/dev/null
	@echo "Finished building training and test utilities!"

init:
	@rm -rf $(RESULT_DIR)
	@mkdir -p $(RESULT_DIR)
	@./bin/trainer epochs 0 workers 0

run:
	@mkdir -p $(RESULT_DIR)
	@echo "-------------------------------------------------------------------" $(OUTPUT)
	@echo "Sessions Started: $(SESSIONS) session(s) with $(EPOCHS) epoch(s) per session." $(OUTPUT)
	@echo "-------------------------------------------------------------------\n" $(OUTPUT)
	@i=1; while [ $$i -le $(SESSIONS) ]; do \
		echo "Session $$i started..." $(OUTPUT); \
		make backup 1>/dev/null 2>&1; \
		echo "Backup Complete!" $(OUTPUT); \
		./bin/trainer epochs $(EPOCHS) batch $(BATCH) alpha $(ALPHA) workers $(WORKERS) break $(BREAK); \
		echo "$(EPOCHS) training epoch(s) completed!" $(OUTPUT); \
		./bin/tester workers $(WORKERS); \
		echo "Testing Complete!" $(OUTPUT); \
		echo "Session $$i completed!\n" $(OUTPUT); \
		i=$$((i+1)); \
	done
	@echo "$(SESSIONS) training session(s) were completed successfully!\n" $(OUTPUT)

epoch:
	@mkdir -p $(RESULT_DIR)
	@echo "Training neural network..."
	@./bin/trainer epochs 1 batch $(BATCH) alpha $(ALPHA) workers $(WORKERS) break $(BREAK)
	@echo "1 training epoch completed!"

eval:
	@mkdir -p $(RESULT_DIR)
	@echo "Testing neural network..."
	@./bin/tester workers $(WORKERS)
	@echo "Testing Complete!"

backup:
	@$(call copy, $(RESULT_DIR), $(BACKUP_DIR))
	@echo "Backup Complete!"

restore:
	@$(call copy, $(BACKUP_DIR), $(RESULT_DIR))
	@rm -rf backup
	@echo "Restoration Complete!"

refresh:
	@rm -f $(RESULT_DIR)/$(ADAMFILE)
	@echo "Training state refreshed successfully!"

reset:
	@rm -rf result
	@echo "Training state has been reset successfully!"

clean:
	@rm -rf build bin lib
	@echo "Finished cleaning build artifacts!"

# -------------------------- private --------------------------

_lib:
	@$(MAKE) -C $(LOGICNN_PROJECT_ROOT) ALIGN=TRUE BUILD_DIR=$(CURDIR)/build/LogicNN LIB_DIR=$(CURDIR)/lib/LogicNN release >/dev/null

bin/trainer: _lib $(TRAINER_OBJ) $(TRAINER_HDR)
	@mkdir -p bin
	@$(CXX) $(TRAINER_OBJ) $(LDFLAGS) -o $@

bin/tester: _lib $(TESTER_OBJ) $(TESTER_HDR)
	@mkdir -p bin
	@$(CXX) $(TESTER_OBJ) $(LDFLAGS) -o $@

build/trainer/%.o: src/%.cpp $(TRAINER_HDR)
	@mkdir -p build/trainer
	@$(CXX) $(CXXFLAGS) -DEMBEDDING_DIM=$(EMBEDDING) -DLOGFILE=\"$(RESULT_DIR)/$(TRAINLOG)\" -DNNFILE=\"$(RESULT_DIR)/$(NNFILE)\" -DADAMFILE=\"$(RESULT_DIR)/$(ADAMFILE)\" -DDATASET_DIR=\"$(TRAINSET_DIR)\" -c $< -o $@

build/tester/%.o: src/%.cpp $(TESTER_HDR)
	@mkdir -p build/tester
	@$(CXX) $(CXXFLAGS) -DEMBEDDING_DIM=$(EMBEDDING) -DLOGFILE=\"$(RESULT_DIR)/$(TESTLOG)\" -DNNFILE=\"$(RESULT_DIR)/$(NNFILE)\" -DDATASET_DIR=\"$(TESTSET_DIR)\" -c $< -o $@

.PRECIOUS: build/trainer/%.o build/tester/%.o build/LogicNN/release/%.o
.PHONY: help build rebuild init run epoch eval backup restore refresh reset clean