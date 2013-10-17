#ds compiler
CC = nvcc

#ds compiler flags
CFLAGS = -c

#ds default field
all: main

	$(CC) bin/main.o -o bin/nbody_cpu

#ds object files
main:

	rm -rf bin
	mkdir bin
	$(CC) $(CFLAGS) src/main.cu -o bin/main.o

#ds mark clean as independent
.PHONY: clean

#ds clean command
clean:

	rm -rf bin
