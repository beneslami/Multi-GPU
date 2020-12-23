LEVEL=../..
include $(LEVEL)/Makefile.common

SOURCES=$(wildcard *.cc)

OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/mk-sched/%.o)
CCFLAGS+=-fPIC

DEBUG_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/mk-sched/%.g)
CCDEBUGFLAGS+=-fPIC

DEP_OBJECTS=$(SOURCES:%.cc=$(BUILD_DIR)/launcher/mk-sched/%.d)

#all: makedirs $(OBJECTS) $(DEP_OBJECTS)
#debug: makedirs $(DEBUG_OBJECTS) $(DEP_OBJECTS)

all: makedirs $(OBJECTS)
debug: makedirs $(DEBUG_OBJECTS)

$(BUILD_DIR)/launcher/mk-sched/%.o: %.cc
	$(CXX) $(CCFLAGS) -o $@ -c $<

$(BUILD_DIR)/launcher/mk-sched/%.g: %.cc
	$(CXX) $(CCDEBUGFLAGS) -o $@ -c $<

$(BUILD_DIR)/launcher/mk-sched/%.d: %.cc
	$(CXX) -MM $< | sed 's!^$(<:.cc=.o):!$(<:.cc=.o) $(<:.cc=.d):!' > $@

makedirs:
	if [ ! -d $(BUILD_DIR)/launcher/mk-sched ]; then mkdir -p $(BUILD_DIR)/launcher/mk-sched; fi;

#include $(DEP_OBJECTS)
