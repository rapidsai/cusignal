NVCC	:=nvcc -ccbin g++
CFLAGS	:=-O3 -std=c++14 -DUSE_NVTX
ARCHES	:=-gencode arch=compute_75,code=\"compute_75,sm_75\"
INC_DIR	:=
LIB_DIR	:=
LIBS	:=

SOURCES := reader \

all: $(SOURCES)
.PHONY: all

reader: reader.o
	$(NVCC) --cudart=static $(CFLAGS) ${ARCHES} $^ -o $@ $(LIBS)

reader.o: reader.cu
	$(NVCC) -dc $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES) reader.o
