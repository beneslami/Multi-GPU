LEVEL=..
include $(LEVEL)/Makefile.common

SOURCES=$(wildcard *.cc)
OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/src/%.o)
DEBUG_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/src/%.g)

all: makedirs $(OBJECTS)
	$(MAKE) -C ./cuda-sim
	$(MAKE) -C ./gpgpu-sim
	$(MAKE) -C ./intersim2
	$(MAKE) "CREATE_LIBRARY=1" -C ./ramulator_sim

debug: makedirs $(DEBUG_OBJECTS)
	$(MAKE) debug -C ./cuda-sim
	$(MAKE) debug -C ./gpgpu-sim
	$(MAKE) -C ./intersim2
	$(MAKE) "CREATE_LIBRARY=1" -C ./ramulator_sim

$(BUILD_DIR)/src/%.o: %.cc
	$(CXX) $(CCFLAGS) -fPIC -o $@ -c $<

$(BUILD_DIR)/src/%.g: %.cc
	$(CXX) $(CCDEBUGFLAGS) -fPIC -o $@ -c $<

makedirs:
	if [ ! -d $(BUILD_DIR)/src ]; then mkdir -p $(BUILD_DIR)/src; fi;
