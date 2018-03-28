CFLAGS ?= -O3 -shared -fPIC
CXX ?= g++
GENERAL_PATH=general
UTILS_PATH=utils

all:	libcCovar.so libcCirc.so

clean:
	cd utils && rm -f libcCirc.so cCirc.o
	cd general && rm -f libcCovar.so cCovar.o

# libcCovar.so:  cd general && cCovar.o Makefile
libcCovar.so:  $(GENERAL_PATH)/cCovar.o Makefile
	cd general && ${CXX} -shared -o libcCovar.so cCovar.o

$(GENERAL_PATH)/cCovar.o:  $(GENERAL_PATH)/cCovar.cc Makefile
	cd general && ${CXX} ${CFLAGS} -c cCovar.cc

libcCirc.so:  $(UTILS_PATH)/cCirc.o Makefile
	cd utils && ${CXX} -shared -o libcCirc.so cCirc.o

$(UTILS_PATH)/cCirc.o:  $(UTILS_PATH)/cCirc.cc Makefile
	cd utils && ${CXX} ${CFLAGS} -c cCirc.cc
