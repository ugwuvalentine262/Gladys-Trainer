CC := gcc
AR := ar

BUILD_DIR := build
LIB_DIR := lib

override CFLAGS := -Werror -Wextra -Wall -std=c17 -I./include
override DEBUG_FLAGS := $(CFLAGS) -g -O0 -DDEBUG
override RELEASE_FLAGS := $(CFLAGS) -O2 -O3 -DNDEBUG
override ASAN_FLAGS := $(CFLAGS) -fsanitize=address -g -O0 -DDEBUG

all: release debug asan

release: $(LIB_DIR)/release/liblogicnn.a $(LIB_DIR)/release/liblogicnn_backprop.a
	@echo Finished building Logic Neural Network library in release mode! 

debug: $(LIB_DIR)/debug/liblogicnn.a $(LIB_DIR)/debug/liblogicnn_backprop.a
	@echo Finished building Logic Neural Network library in debug mode! 

asan: $(LIB_DIR)/asan/liblogicnn.a $(LIB_DIR)/asan/liblogicnn_backprop.a
	@echo Finished building Logic Neural Network library with address sanitizer! 

clean:
	@rm -f *.a *.o
	@rm -rf $(BUILD_DIR) $(LIB_DIR)
	@echo Finished cleaning all build artifacts!

$(LIB_DIR)/release/liblogicnn.a: $(BUILD_DIR)/release/logicnn.o
	@mkdir -p "$(LIB_DIR)/release"
	@$(AR) rcs $@ $<

$(LIB_DIR)/release/liblogicnn_backprop.a: $(BUILD_DIR)/release/logicnn_backprop.o
	@mkdir -p "$(LIB_DIR)/release"
	@$(AR) rcs $@ $<

$(LIB_DIR)/debug/liblogicnn.a: $(BUILD_DIR)/debug/logicnn.o
	@mkdir -p "$(LIB_DIR)/debug"
	@$(AR) rcs $@ $<

$(LIB_DIR)/debug/liblogicnn_backprop.a: $(BUILD_DIR)/debug/logicnn_backprop.o
	@mkdir -p "$(LIB_DIR)/debug"
	@$(AR) rcs $@ $<

$(LIB_DIR)/asan/liblogicnn.a: $(BUILD_DIR)/asan/logicnn.o
	@mkdir -p "$(LIB_DIR)/asan"
	@$(AR) rcs $@ $<

$(LIB_DIR)/asan/liblogicnn_backprop.a: $(BUILD_DIR)/asan/logicnn_backprop.o
	@mkdir -p "$(LIB_DIR)/asan"
	@$(AR) rcs $@ $<

$(BUILD_DIR)/release/logicnn.o: ./src/logicnn.c ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/release"
	@$(CC) $(RELEASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/release/logicnn_backprop.o: ./src/logicnn_backprop.c ./include/logicnn_backprop.h ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/release"
	@$(CC) $(RELEASE_FLAGS) -c $< -o $@

$(BUILD_DIR)/debug/logicnn.o: ./src/logicnn.c ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/debug"
	@$(CC) $(DEBUG_FLAGS) -c $< -o $@

$(BUILD_DIR)/debug/logicnn_backprop.o: ./src/logicnn_backprop.c ./include/logicnn_backprop.h ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/debug"
	@$(CC) $(DEBUG_FLAGS) -c $< -o $@

$(BUILD_DIR)/asan/logicnn.o: ./src/logicnn.c ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/asan"
	@$(CC) $(ASAN_FLAGS) -c $< -o $@

$(BUILD_DIR)/asan/logicnn_backprop.o: ./src/logicnn_backprop.c ./include/logicnn_backprop.h ./include/logicnn.h
	@mkdir -p "$(BUILD_DIR)/asan"
	@$(CC) $(ASAN_FLAGS) -c $< -o $@

.PRECIOUS: $(BUILD_DIR)/release/%.o $(BUILD_DIR)/debug/%.o $(BUILD_DIR)/asan/%.o
.PHONY: all clean release debug asan