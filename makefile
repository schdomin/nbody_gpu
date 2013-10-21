#ds compiler
CC = nvcc

#ds compiler flags
CFLAGS = -c

#ds default field
all: main

	$(CC) bin/CCubicDomain.o bin/main.o -o bin/nbody_gpu

#ds object files
main:

	rm -rf bin
	mkdir bin
	$(CC) $(CFLAGS) src/CCubicDomain.cu -o bin/CCubicDomain.o
	$(CC) $(CFLAGS) src/main.cu -o bin/main.o

#ds mark clean as independent
.PHONY: clean

#ds clean command
clean:

	rm -rf bin
