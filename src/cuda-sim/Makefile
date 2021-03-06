LEVEL=../..
include $(LEVEL)/Makefile.common

CPP = $(CXX)
OUTPUT_DIR=$(BUILD_DIR)/src/cuda-sim

OPT := -O2 -Wall -Wno-unused-function -Wno-sign-compare -I$(CUDA_INSTALL_PATH)/include/ -I$(OUTPUT_DIR)/ -I.
OPT += -fPIC 
OPT += -DCUDART_VERSION=$(CUDART_VERSION)
CXX_OPT = $(OPT)
CXX_OPT += -std=c++0x

DEBUG_OPT := -O0 -g -Wall -Wno-unused-function -Wno-sign-compare -I$(CUDA_INSTALL_PATH)/include/ -I$(OUTPUT_DIR)/ -I. -fPIC -DCUDART_VERSION=$(CUDART_VERSION)
DEBUG_CXX_OPT = $(DEBUG_OPT)
DEBUG_CXX_OPT += -std=c++0x

OBJS := $(OUTPUT_DIR)/ptx_parser.o $(OUTPUT_DIR)/ptx_loader.o $(OUTPUT_DIR)/cuda_device_printf.o $(OUTPUT_DIR)/instructions.o $(OUTPUT_DIR)/cuda-sim.o $(OUTPUT_DIR)/ptx_ir.o $(OUTPUT_DIR)/ptx_sim.o  $(OUTPUT_DIR)/memory.o $(OUTPUT_DIR)/ptx-stats.o $(OUTPUT_DIR)/decuda_pred_table/decuda_pred_table.o $(OUTPUT_DIR)/ptx.tab.o $(OUTPUT_DIR)/lex.ptx_.o $(OUTPUT_DIR)/ptxinfo.tab.o $(OUTPUT_DIR)/lex.ptxinfo_.o
DEBUG_OBJS := $(OUTPUT_DIR)/ptx_parser.g $(OUTPUT_DIR)/ptx_loader.g $(OUTPUT_DIR)/cuda_device_printf.g $(OUTPUT_DIR)/instructions.g $(OUTPUT_DIR)/cuda-sim.g $(OUTPUT_DIR)/ptx_ir.g $(OUTPUT_DIR)/ptx_sim.g  $(OUTPUT_DIR)/memory.g $(OUTPUT_DIR)/ptx-stats.g $(OUTPUT_DIR)/decuda_pred_table/decuda_pred_table.g $(OUTPUT_DIR)/ptx.tab.g $(OUTPUT_DIR)/lex.ptx_.g $(OUTPUT_DIR)/ptxinfo.tab.g $(OUTPUT_DIR)/lex.ptxinfo_.g

all: makedirs $(OBJS)

debug: makedirs $(DEBUG_OBJS)

$(OUTPUT_DIR)/ptx.tab.o: $(OUTPUT_DIR)/ptx.tab.c
	$(CPP) -c $(OPT) -DYYDEBUG $(OUTPUT_DIR)/ptx.tab.c -o $(OUTPUT_DIR)/ptx.tab.o

$(OUTPUT_DIR)/lex.ptx_.o: $(OUTPUT_DIR)/lex.ptx_.c 
	$(CPP) -c $(OPT) $(OUTPUT_DIR)/lex.ptx_.c -o $(OUTPUT_DIR)/lex.ptx_.o

$(OUTPUT_DIR)/ptxinfo.tab.o: $(OUTPUT_DIR)/ptxinfo.tab.c
	$(CPP) -c $(OPT) -DYYDEBUG $(OUTPUT_DIR)/ptxinfo.tab.c -o $(OUTPUT_DIR)/ptxinfo.tab.o

$(OUTPUT_DIR)/lex.ptxinfo_.o: $(OUTPUT_DIR)/lex.ptxinfo_.c $(OUTPUT_DIR)/ptxinfo.tab.c
	$(CPP) -c $(OPT) $(OUTPUT_DIR)/lex.ptxinfo_.c -o $(OUTPUT_DIR)/lex.ptxinfo_.o

$(OUTPUT_DIR)/ptx.tab.c: ptx.y
	bison --name-prefix=ptx_ -v -d ptx.y --file-prefix=$(OUTPUT_DIR)/ptx

$(OUTPUT_DIR)/ptxinfo.tab.c: ptxinfo.y
	bison --name-prefix=ptxinfo_ -v -d ptxinfo.y --file-prefix=$(OUTPUT_DIR)/ptxinfo

$(OUTPUT_DIR)/lex.ptx_.c: ptx.l
	flex --outfile=$(OUTPUT_DIR)/lex.ptx_.c ptx.l 

$(OUTPUT_DIR)/lex.ptxinfo_.c: ptxinfo.l
	flex --outfile=$(OUTPUT_DIR)/lex.ptxinfo_.c ptxinfo.l 

clean:
	rm -f *~ *.o *.gcda *.gcno *.gcov libgpgpu_ptx_sim.a \
		ptx.tab.h ptx.tab.c ptx.output lex.ptx_.c \
		ptxinfo.tab.h ptxinfo.tab.c ptxinfo.output lex.ptxinfo_.c \
		instructions.h ptx_parser_decode.def directed_tests.log 
	rm -f $(OUTPUT_DIR)/decuda_pred_table/*.o

$(OUTPUT_DIR)/%.o: %.c
	$(CPP) -c $(OPT) $< -o $(OUTPUT_DIR)/$*.o
$(OUTPUT_DIR)/%.o: %.cc
	$(CPP) -c $(CXX_OPT) $< -o $(OUTPUT_DIR)/$*.o

$(OUTPUT_DIR)/ptx.tab.g: $(OUTPUT_DIR)/ptx.tab.c
	$(CPP) -c $(DEBUG_OPT) -DYYDEBUG $(OUTPUT_DIR)/ptx.tab.c -o $(OUTPUT_DIR)/ptx.tab.g

$(OUTPUT_DIR)/lex.ptx_.g: $(OUTPUT_DIR)/lex.ptx_.c 
	$(CPP) -c $(DEBUG_OPT) $(OUTPUT_DIR)/lex.ptx_.c -o $(OUTPUT_DIR)/lex.ptx_.g

$(OUTPUT_DIR)/ptxinfo.tab.g: $(OUTPUT_DIR)/ptxinfo.tab.c
	$(CPP) -c $(DEBUG_OPT) -DYYDEBUG $(OUTPUT_DIR)/ptxinfo.tab.c -o $(OUTPUT_DIR)/ptxinfo.tab.g

$(OUTPUT_DIR)/lex.ptxinfo_.g: $(OUTPUT_DIR)/lex.ptxinfo_.c $(OUTPUT_DIR)/ptxinfo.tab.c
	$(CPP) -c $(DEBUG_OPT) $(OUTPUT_DIR)/lex.ptxinfo_.c -o $(OUTPUT_DIR)/lex.ptxinfo_.g

$(OUTPUT_DIR)/%.g: %.c
	$(CPP) -c $(DEBUG_OPT) $< -o $(OUTPUT_DIR)/$*.g
$(OUTPUT_DIR)/%.g: %.cc
	$(CPP) -c $(DEBUG_CXX_OPT) $< -o $(OUTPUT_DIR)/$*.g

$(OUTPUT_DIR)/instructions.h: instructions.cc
	@touch $*.h
	@chmod +w $*.h
	@echo "// DO NOT EDIT THIS FILE! IT IS AUTOMATICALLY GENERATED BY THE MAKEFILE (see target for instructions.h)" > $*.h
	@echo "#include \"ptx_ir.h\"" >> $*.h
	@echo "#ifndef instructions_h_included" >> $*.h
	@echo "#define instructions_h_included" >> $*.h
	@cat $< | grep "_impl(" | sed 's/{.*//' | sed 's/$$/;/' >> $*.h
	@echo "#endif" >> $*.h
	@chmod -w $*.h
	@echo "created $(OUTPUT_DIR)/instructions.h"

$(OUTPUT_DIR)/ptx_parser_decode.def: $(OUTPUT_DIR)/ptx.tab.c
ifeq ($(shell uname),Linux)
	cat $(OUTPUT_DIR)/ptx.tab.h | grep "=" | sed 's/^[ ]\+//' | sed 's/[=,]//g' | sed 's/\([_A-Z1-9]\+\)[ ]\+\([0-9]\+\)/\1 \1/' | sed 's/^/DEF(/' | sed 's/ /,"/' | sed 's/$$/")/' > $(OUTPUT_DIR)/ptx_parser_decode.def
else
	cat $(OUTPUT_DIR)/ptx.tab.h | grep "=" | sed -E 's/^ +//' | sed 's/[=,]//g' | sed -E 's/([_A-Z1-9]+).*/\1 \1/' | sed 's/^/DEF(/' | sed 's/ /,"/' | sed 's/$$/")/' > $(OUTPUT_DIR)/ptx_parser_decode.def
endif

$(OUTPUT_DIR)/instructions.o: $(OUTPUT_DIR)/instructions.h $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/cuda_device_printf.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx_ir.o: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptx_loader.o: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptx_parser.o: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptxinfo.tab.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx-stats.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx_sim.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/cuda-sim.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/lex.ptxinfo_.o: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/lex.ptx_.o: $(OUTPUT_DIR)/ptx.tab.c

$(OUTPUT_DIR)/instructions.g: $(OUTPUT_DIR)/instructions.h $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/cuda_device_printf.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx_ir.g: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptx_loader.g: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptx_parser.g: $(OUTPUT_DIR)/ptx.tab.c $(OUTPUT_DIR)/ptx_parser_decode.def
$(OUTPUT_DIR)/ptxinfo.tab.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx-stats.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/ptx_sim.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/cuda-sim.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/lex.ptxinfo_.g: $(OUTPUT_DIR)/ptx.tab.c
$(OUTPUT_DIR)/lex.ptx_.g: $(OUTPUT_DIR)/ptx.tab.c

makedirs:
	if [ ! -d $(BUILD_DIR)/src/cuda-sim/decuda_pred_table ]; then mkdir -p $(BUILD_DIR)/src/cuda-sim/decuda_pred_table; fi;
