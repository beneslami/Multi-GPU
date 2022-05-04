LEVEL=../..
include $(LEVEL)/Makefile.common

CXX = g++-5
CC = gcc-5
DEFINE = -DCREATE_LIBRARY
INCPATH = -I. -Iarbiters -Iallocators -Irouters -Inetworks -Ipower -I../gpgpu-sim/
CPPFLAGS = -Wall $(INCPATH) -O2 -fPIC $(DEFINE)

LEX = flex
YACC = bison -y

OBJDIR := $(BUILD_DIR)/src/intersim2

# simulator source files
CPP_SRCS =  \
   config_utils.cpp \
   booksim_config.cpp \
   module.cpp \
   buffer.cpp \
   vc.cpp \
   routefunc.cpp \
   traffic.cpp \
   flitchannel.cpp \
   trafficmanager.cpp \
   batchtrafficmanager.cpp \
   packet_reply_info.cpp \
   buffer_state.cpp \
   stats.cpp \
   credit.cpp \
   outputset.cpp \
   flit.cpp \
   injection.cpp\
   misc_utils.cpp\
   rng_wrapper.cpp\
   rng_double_wrapper.cpp\
   power_module.cpp \
   switch_monitor.cpp \
   buffer_monitor.cpp \
   main.cpp \
   gputrafficmanager.cpp \
   intersim_config.cpp \
   interconnect_interface.cpp

LEX_OBJS  = ${OBJDIR}/lex.yy.o
YACC_OBJS = ${OBJDIR}/y.tab.o

# networks 
NETWORKS:= $(wildcard networks/*.cpp) 
ALLOCATORS:= $(wildcard allocators/*.cpp)
ARBITERS:= $(wildcard arbiters/*.cpp)
ROUTERS:= $(wildcard routers/*.cpp)
POWER:= $(wildcard power/*.cpp)

#--- Make rules ---
OBJS :=  $(LEX_OBJS) $(YACC_OBJS)\
 $(CPP_SRCS:%.cpp=${OBJDIR}/%.o)\
 $(NETWORKS:networks/%.cpp=${OBJDIR}/%.o)\
 $(ALLOCATORS:allocators/%.cpp=${OBJDIR}/%.o)\
 $(ARBITERS:arbiters/%.cpp=${OBJDIR}/%.o)\
 $(ROUTERS:routers/%.cpp=${OBJDIR}/%.o)\
 $(POWER:power/%.cpp=${OBJDIR}/%.o)

.PHONY: clean

all: makedirs $(OBJS)

# rules to compile simulator
${LEX_OBJS}: lex.yy.c y.tab.h
	$(CC) $(CPPFLAGS) -c $< -o $@

${YACC_OBJS}: y.tab.c y.tab.h
	$(CC) $(CPPFLAGS) -c $< -o $@

${OBJDIR}/%.o: %.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile networks
${OBJDIR}/%.o: networks/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile arbiters
${OBJDIR}/%.o: arbiters/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile allocators
${OBJDIR}/%.o: allocators/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile routers
${OBJDIR}/%.o: routers/%.cpp 
	$(CXX) $(CPPFLAGS) -c $< -o $@

# rules to compile power classes
${OBJDIR}/%.o: power/%.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) 
	rm -f *~
	rm -f allocators/*~
	rm -f arbiters/*~
	rm -f networks/*~
	rm -f runfiles/*~
	rm -f routers/*~
	rm -f examples/*~
	rm -f y.tab.c y.tab.h lex.yy.c
	rm -f moc_bgui.cpp

y.tab.c y.tab.h: config.y
	$(YACC) -d $<

lex.yy.c: config.l
	$(LEX) $<

makedirs:
	if [ ! -d $(BUILD_DIR)/src/intersim2 ]; then mkdir -p $(BUILD_DIR)/src/intersim2; fi;

# DO NOT DELETE
