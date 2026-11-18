# Makefile  –  Windows (MinGW/MSYS2) hoặc Linux/macOS
# Sử dụng: make  hoặc  make run

GPP     ?= g++
CXXFLAGS = -std=c++17 -O2 -march=native -Wall -Wextra
OMP_FLAG = -fopenmp
SRC      = src/cpu/cpu_mc.cpp
BIN      = cpu_mc
ROOT     = $(CURDIR)

.PHONY: all run clean

all: $(BIN)

$(BIN): $(SRC)
	$(GPP) $(CXXFLAGS) $(OMP_FLAG) -o $@ $^
	@echo "Build OK → $(BIN)"

run: $(BIN)
	./$(BIN) "$(ROOT)"

clean:
	rm -f $(BIN) $(BIN).exe
