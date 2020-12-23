LEVEL=..
include $(LEVEL)/Makefile.common

SOURCES=$(wildcard *.cc)
OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/src/%.o)
DEBUG_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/src/%.g)

all: makedirs $(OBJECTS)
	make -C ./cuda-sim
	make -C ./gpgpu-sim
	make -C ./intersim2
	make "CREATE_LIBRARY=1" -C ./ramulator_sim

debug: makedirs $(DEBUG_OBJECTS)
	make debug -C ./cuda-sim
	make debug -C ./gpgpu-sim
	make -C ./intersim2
	make "CREATE_LIBRARY=1" -C ./ramulator_sim

$(BUILD_DIR)/src/%.o: %.cc
	$(CXX) $(CCFLAGS) -fPIC -o $@ -c $<

$(BUILD_DIR)/src/%.g: %.cc
	$(CXX) $(CCDEBUGFLAGS) -fPIC -o $@ -c $<

makedirs:
	if [ ! -d $(BUILD_DIR)/src ]; then mkdir -p $(BUILD_DIR)/src; fi;
