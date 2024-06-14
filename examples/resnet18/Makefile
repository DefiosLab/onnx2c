CC=gcc
CFLAGS=-fPIC -std=c99 -shared
INCLUDES=-I../../clang_operators/ -I./artifacts/
SOURCES=wrapper.c artifacts/inference.c ../../clang_operators/operator.c
OUTPUT=resnet.so

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) -o $(OUTPUT)

clean:
	rm -f $(OUTPUT)
