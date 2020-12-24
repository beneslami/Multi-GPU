LEVEL=../..
include $(LEVEL)/Makefile.common

CPP = $(CXX)
OEXT = o
GEXT = g
OUTPUT_DIR=$(BUILD_DIR)/src/gpgpu-sim

CXXFLAGS = -Wall -std=c++0x
OPTFLAGS = -O2 -fPIC
POWER_FLAGS= -I../gpuwattch/

DEBUG_OPTFLAGS = -O0 -g -fPIC

SRCS = $(shell ls *.cc)
#EXCLUDES += power_interface.cc
EXCLUDES += 
CSRCS = $(filter-out $(EXCLUDES), $(SRCS))

OBJS = $(CSRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))
DEBUG_OBJS = $(CSRCS:%.cc=$(OUTPUT_DIR)/%.$(GEXT))
 
all: makedirs $(OBJS)

debug: makedirs $(DEBUG_OBJS)

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $(OUTPUT_DIR)/$*.$(OEXT) -c $*.cc

$(OUTPUT_DIR)/%.$(GEXT): %.cc
	$(CPP) $(DEBUG_OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $(OUTPUT_DIR)/$*.$(GEXT) -c $*.cc

clean:
	rm -f *.o core *~ *.a 
	rm -f Makefile.makedepend Makefile.makedepend.bak

$(OUTPUT_DIR)/option_parser.$(OEXT): option_parser.h

$(OUTPUT_DIR)/dram_sched.$(OEXT): $(OUTPUT_DIR)/../cuda-sim/ptx.tab.h

$(OUTPUT_DIR)/../cuda-sim/ptx.tab.h:
	$(MAKE) -C ../cuda-sim/ $(OUTPUT_DIR)/../cuda-sim/ptx.tab.c

makedirs:
	if [ ! -d $(BUILD_DIR)/src/gpgpu-sim ]; then mkdir -p $(BUILD_DIR)/src/gpgpu-sim; fi;
