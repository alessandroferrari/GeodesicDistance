all: 
	g++ -I . -o geodesic_distance wdtocs.cpp dtocs.cpp main.cpp `pkg-config opencv --cflags --libs`
