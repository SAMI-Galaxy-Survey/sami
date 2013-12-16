CFLAGS ?= -O3 -shared -fPIC
CXX ?= g++

all:	libcCirc.so

clean:
	rm -f libcCirc.so cCirc.o

libcCirc.so:  cCirc.o Makefile
	${CXX} -shared -o libcCirc.so cCirc.o

cCirc.o:  cCirc.cc Makefile
	${CXX} ${CFLAGS} -c cCirc.cc
