include make.inc

BUILD_DIR := build
SRC_DIR := ..
TEST_SRC_DIR := src
CUDA_SRC_DIR := ..

SRCS := $(wildcard ${SRC_DIR}/*.cpp)
OBJS := $(patsubst  ${SRC_DIR}/%.cpp, ${BUILD_DIR}/%.o, $(SRCS))
TEST_SRCS := $(wildcard ${TEST_SRC_DIR}/*.cpp)
TEST_OBJS := $(patsubst  ${TEST_SRC_DIR}/%.cpp, ${BUILD_DIR}/%.o, $(TEST_SRCS))
SRCS += $(TEST_SRCS)
OBJS += $(TEST_OBJS)

CUDA_SRCS := $(wildcard ${CUDA_SRC_DIR}/*.cu)
CUDA_OBJS := $(patsubst  ${CUDA_SRC_DIR}/%.cu, ${BUILD_DIR}/%.o, $(CUDA_SRCS))

# print:
# 	@echo "SRCS: $(SRCS)"
# 	@echo "OBJS: $(OBJS)"
# 	@echo "TEST_SRCS: $(TEST_SRCS)"
# 	@echo "TEST_OBJS: $(TEST_OBJS)"
# 	@echo "CUDA_SRCS: $(CUDA_SRCS)"
# 	@echo "CUDA_OBJS: $(CUDA_OBJS)"


all: ${BUILD_DIR} $(OBJS) $(CXX_OBJS) $(CUDA_OBJS)
	$(CC) $(CXXFLAGS) $(OBJS) $(CXX_OBJS) $(CUDA_OBJS) $(LIBS) -lm -lgtest -o ${BUILD_DIR}/test

$(BUILD_DIR)/%.o: ${SRC_DIR}/%.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
$(BUILD_DIR)/%.o: ${TEST_SRC_DIR}/%.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: ${CUDA_SRC_DIR}/%.cu
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDES) -c $< -o $@ 

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:	
	rm -fr build/

# PHONY targets
.PHONY: all clean
