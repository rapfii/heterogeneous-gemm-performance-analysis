CC       := gcc
CFLAGS   := -O3 -Wall -Wextra -std=c11
LDFLAGS  := -fopenmp -lOpenCL -lm

TARGET   := gemm_runner

SRCS     := src/main.c \
            src/sequential.c \
            src/openmp.c \
            src/opencl/opencl_host.c

.PHONY: build clean benchmark graphs all

build: $(TARGET)

$(TARGET): $(SRCS) src/gemm_common.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf results test

benchmark: $(TARGET)
	@bash scripts/benchmark.sh

graphs:
	@python3 scripts/generate_graphs.py

all: build benchmark graphs
