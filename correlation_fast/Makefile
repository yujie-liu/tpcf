SHELL    = /bin/sh
CXX      = g++
CXXFLAGS = $(shell root-config --cflags)
LD       = g++
LDFLAGS  = $(shell root-config --libs)

.PHONY: all clean

all: correlation_fast step2 tab_integral

correlation_fast: correlation_fast.o ConfigParser.o
	$(LD) $(LDFLAGS) $^ -o $@

step2: step2.o ConfigParser.o
	$(LD) $(LDFLAGS) $^ -o $@

tab_integral: tab_integral.o ConfigParser.o
	$(LD) $(LDFLAGS) $^ -o $@

%.o:%.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@

clean:
	rm -f *.o a.out
