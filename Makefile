LEVEL=.
include $(LEVEL)/Makefile.common

TARGETS=$(BIN_DIR)/multikernel-sim
OBJECTS=$(wildcard $(BUILD_DIR)/launcher/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/launcher/mk-sched/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/launcher/stats/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/common/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/cuda-sim/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/cuda-sim/decuda_pred_table/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/gpgpu-sim/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/intersim2/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/ramulator_sim/*.o)
OBJECTS+=$(wildcard $(BUILD_DIR)/src/*.o)
OBJECTS+=$(wildcard ./kain_gpuwattch/*.o)

DEBUG_TARGETS=$(BIN_DIR)/multikernel-sim.debug
DEBUG_OBJECTS=$(wildcard $(BUILD_DIR)/launcher/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/launcher/mk-sched/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/launcher/stats/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/common/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/cuda-sim/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/cuda-sim/decuda_pred_table/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/gpgpu-sim/*.g)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/intersim2/*.o)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/ramulator_sim/*.o)
DEBUG_OBJECTS+=$(wildcard $(BUILD_DIR)/src/*.g)

all: subdirs $(TARGETS)

debug: subdirs_debug $(DEBUG_TARGETS)

$(BIN_DIR)/multikernel-sim: makedirs $(OBJECTS)
	$(CXX) $(CCFLAGS) $(OBJECTS) -lz -lm -lpthread -o $@

$(BIN_DIR)/multikernel-sim.debug: makedirs $(DEBUG_OBJECTS)
	$(CXX) $(CCDEBUGFLAGS) $(DEBUG_OBJECTS) -lz -lm -lpthread -o $@

subdirs:
	make -C ./common
	make -C ./src
	make -C ./launcher
	make -C ./libcuda

subdirs_debug:
	make debug -C ./common
	make debug -C ./src
	make debug -C ./launcher
	make debug -C ./libcuda

.PHONY: clean
clean:
	rm -rf build/
	rm -rf lib/
	rm -rf bin/

makedirs:
	if [ ! -d $(BIN_DIR) ]; then mkdir -p $(BIN_DIR); fi;
