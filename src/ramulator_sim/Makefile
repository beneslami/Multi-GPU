LEVEL=../..
include $(LEVEL)/Makefile.common

#CXX = g++
#CC = gcc
CXX=/usr/bin/g++-5
CC=/usr/bin/gcc-5
CREATE_LIBRARY ?= 0
INTERFACE = gpu_wrapper.cpp
DEBUG ?= 0

ifeq ($(CREATE_LIBRARY),1)
INCPATH += -I$(GPGPUSIM_ROOT)/src/gpgpu-sim/
endif

CPPFLAGS += -Wall $(INCPATH)
ifneq ($(DEBUG),1)
CPPFLAGS += -O3
endif
CPPFLAGS += -g
CPPFLAGS += -fPIC
CPPFLAGS += -std=c++11
LFLAGS +=

ifeq ($(SIM_OBJ_FILES_DIR),)
OBJDIR := obj
else
OBJDIR := $(SIM_OBJ_FILES_DIR)/ramulator_sim
endif
PROG   := ramulator_sim
OBJDIR := $(BUILD_DIR)/src/ramulator_sim

CPP_SRCS = $(wildcard *.cpp)

OBJS := $(CPP_SRCS:%.cpp=${OBJDIR}/%.o)

.PHONY: clean

ifeq ($(CREATE_LIBRARY),1)
all: makedirs $(OBJS)
else
all:$(PROG)
$(PROG): $(OBJS)
	 $(CXX) $(LFLAGS) $^ -o $@
endif
${OBJDIR}/%.o: %.cpp
	$(CXX) $(CPPFLAGS) -DRAMULATOR -c $< -o $@

clean:
	rm -f $(OBJS) 
	rm -f $(PROG)
	rm -f *~

makedirs:
	if [ ! -d $(BUILD_DIR)/src/ramulator_sim ]; then mkdir -p $(BUILD_DIR)/src/ramulator_sim; fi;
