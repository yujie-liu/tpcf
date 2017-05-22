SHELL    = /bin/sh
CXX      = g++
CXXFLAGS = $(shell root-config --cflags)
LD       = g++
LDFLAGS  = $(shell root-config --libs)

.PHONY: all clean

all: correlation_fast step2

correlation_fast: correlation_fast.o ConfigParser.o
	$(LD) $(LDFLAGS) $^ -o $@

step2: step2.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o:%.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm -f *.o a.out
