LEVEL=..
include $(LEVEL)/Makefile.common

SOURCES=$(wildcard *.cc)

OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/%.o)
CCFLAGS+=-fPIC

DEBUG_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/%.g)
CCDEBUGFLAGS+=-fPIC

DEP_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/%.d)

all: makedirs $(OBJECTS)
	make -C ./mk-sched
	make -C ./stats

debug: makedirs $(DEBUG_OBJECTS)
	make debug -C ./mk-sched
	make debug -C ./stats

$(BUILD_DIR)/launcher/%.o: %.cc
	$(CXX) $(CCFLAGS) -o $@ -c $<

$(BUILD_DIR)/launcher/%.g: %.cc
	$(CXX) $(CCDEBUGFLAGS) -o $@ -c $<

$(BUILD_DIR)/launcher/%.d: %.cc
	$(CXX) -MM $< | sed 's!^$(<:.cc=.o):!$(<:.cc=.o) $(<:.cc=.d):!' > $@

makedirs:
	if [ ! -d $(BUILD_DIR)/launcher ]; then mkdir -p $(BUILD_DIR)/launcher; fi;

#include $(DEP_OBJECTS)
