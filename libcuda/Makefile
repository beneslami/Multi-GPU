LEVEL=..
include $(LEVEL)/Makefile.common

TARGETS=$(LIB_DIR)/libcudart.so
DEP_OBJECTS=$(BUILD_DIR)/libcuda/cuda_runtime_api.d

#all: makedirs $(TARGETS) $(DEP_OBJECTS)
#debug: makedirs $(TARGETS) $(DEP_OBJECTS)

all: makedirs $(TARGETS)
debug: makedirs $(TARGETS)

$(LIB_DIR)/libcudart.so: $(BUILD_DIR)/libcuda/cuda_runtime_api.o $(BUILD_DIR)/launcher/communicate.o $(BUILD_DIR)/common/cu_event_st.o $(LEVEL)/kain_cuobdump_obj/*.o
	$(CXX) -shared -Wl,-soname,libcudart.so \
			$(BUILD_DIR)/libcuda/cuda_runtime_api.o \
			$(BUILD_DIR)/launcher/communicate.o \
			$(BUILD_DIR)/common/cu_event_st.o \
			$(LEVEL)/kain_cuobdump_obj/*.o \
			-lm -lz -lGL -pthread \
			-o $(LIB_DIR)/libcudart.so
	if [ ! -f $(LIB_DIR)/libcudart.so.2 ]; then ln -s libcudart.so $(LIB_DIR)/libcudart.so.2; fi
	if [ ! -f $(LIB_DIR)/libcudart.so.3 ]; then ln -s libcudart.so $(LIB_DIR)/libcudart.so.3; fi
	if [ ! -f $(LIB_DIR)/libcudart.so.4 ]; then ln -s libcudart.so $(LIB_DIR)/libcudart.so.4; fi

$(BUILD_DIR)/libcuda/cuda_runtime_api.o: cuda_runtime_api.cc
	$(CXX) $(CCFLAGS) -fPIC -o $@ -c $<

$(BUILD_DIR)/libcuda/cuda_runtime_api.d: cuda_runtime_api.cc
	$(CXX) -MM $< | sed 's!^$(<:.cc=.o):!$(<:.cc=.o) $(<:.cc=.d):!' > $@

makedirs:
	if [ ! -d $(LIB_DIR) ]; then mkdir -p $(LIB_DIR); fi;
	if [ ! -d $(BUILD_DIR) ]; then mkdir -p $(BUILD_DIR); fi;
	if [ ! -d $(BUILD_DIR)/libcuda ]; then mkdir -p $(BUILD_DIR)/libcuda; fi;

#include $(DEP_OBJECTS)
