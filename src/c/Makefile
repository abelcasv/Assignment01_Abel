CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11
TARGET  = benchmark.exe
LIBS    = -lpsapi

.PHONY: all clean

all: $(TARGET)

$(TARGET): benchmark.c
	$(CC) $(CFLAGS) benchmark.c -o $(TARGET) $(LIBS)

clean:
	del /f $(TARGET)