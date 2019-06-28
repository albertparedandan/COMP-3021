CXX=g++8
CXXFLAGS=-std=c++17 -I../include
LDIR=../lib
LIBS=-lmake_transducer

main: main.cpp assignment.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(LDIR) $(LIBS)

assignment.o: assignment.cpp assignment.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean cleanest

clean:
	rm -f *.o *~ core

cleanest:
	rm -f *.o *~ core predict.xml model.xml
