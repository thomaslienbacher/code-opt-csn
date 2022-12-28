CC=g++
OBJDUMP_FLAGS=--no-addresses -Cd -j .text

all: 
	$(CC) -c -O3 -march=native example.cpp -o example_native.o
	$(CC) -c -O3 -msse4 example.cpp -o example_sse4.o
	$(CC) -c -O3 -mavx2 example.cpp -o example_avx2.o
	$(CC) -c -O3 -march=x86-64 example.cpp -o example_generic.o


view:
	objdump $(OBJDUMP_FLAGS) example_native.o > example_native.dump
	objdump $(OBJDUMP_FLAGS) example_sse4.o > example_sse4.dump
	objdump $(OBJDUMP_FLAGS) example_avx2.o > example_avx2.dump
	objdump $(OBJDUMP_FLAGS) example_generic.o > example_generic.dump

clean:
	rm -v *.o 
	rm -v *.dump
