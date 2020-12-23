LEVEL=..
include $(LEVEL)/Makefile.common

SOURCES=$(wildcard *.cc)

OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/common/%.o)
CCFLAGS+=-fPIC

DEBUG_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/common/%.g)
CCDEBUGFLAGS+=-fPIC

DEP_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/common/%.d)

#all: makedirs $(OBJECTS) $(DEP_OBJECTS)
#debug: makedirs $(DEBUG_OBJECTS) $(DEP_OBJECTS)

all: makedirs $(OBJECTS)
debug: makedirs $(DEBUG_OBJECTS)

$(BUILD_DIR)/common/%.o: %.cc
	$(CXX) $(CCFLAGS) -o $@ -c $<

$(BUILD_DIR)/common/%.g: %.cc
	$(CXX) $(CCDEBUGFLAGS) -o $@ -c $<

$(BUILD_DIR)/common/%.d: %.cc
	$(CXX) -MM $< | sed 's!^$(<:.cc=.o):!$(<:.cc=.o) $(<:.cc=.d):!' > $@

makedirs:
	if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi;
	if [ ! -d $(BUILD_DIR)/common ]; then mkdir -p $(BUILD_DIR)/common; fi;

#include $(DEP_OBJECTS)
