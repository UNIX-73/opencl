
ifndef CC
	CC = gcc
endif

INCLUDE = $(COMMON_DIR)

LINK = -lOpenCL -lm

CCFLAGS = $(LINK) -O3 -I$(INCLUDE)

LIBS = -fopenmp

COMMON_DIR = ./C_common

C_COMM_FILES := $(shell find $(COMMON_DIR) -type f -name '*.c')

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
       LIBS =  
endif

pi: pi.c $(C_COMM_FILES)
	$(CC) $^ $(CCFLAGS) -o $@

clean:
	rm -f pi
