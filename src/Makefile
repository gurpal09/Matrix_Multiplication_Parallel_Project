SHELL = /bin/sh

IDIR =../include
ODIR=../obj
BDIR=../bin
BKPDIR=../bkp

PRG = project2
_PRG = $(PRG).exe
EXECUTABLE = $(_PRG:%=$(BDIR)/%)
NOW=$(shell date +"%d-%b-%y")
_TAR:= project2_$(NOW)
TAR=$(patsubst %,$(BKPDIR)/%,$(_TAR))

CC=nvcc
DEBUG = -g
CFLAGS =  -O3 -arch=sm_30  $(DEBUG) -I$(IDIR)
LDFLAGS =
LIBS = -L/cm/shared/apps/gsl/2.3/lib -lgsl -lgslcblas -lcublas
SOURCES = cpu_matrix_multiply.cu cpu_ddot.cu cpu_daxpy.cu main.cu\
          gpu_matrix_multiply.cu gpu_ddot.cu gpu_daxpy.cu

_DEPS = cpumatrixmultiply.h cpuddot.h cpudaxpy.h \
          gpumatrixmultiply.h gpuddot.h gpudaxpy.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))


_OBJ = $(SOURCES:.cu=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) 
	
$(EXECUTABLE): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(EXECUTABLE) core

.PHONY: tar
tar:
	tar -cvf $(TAR).tar $(SOURCES) $(DEPS) Makefile
	gzip $(TAR).tar
