CC=gcc
CFLAGS=-O3
CLIBS=-lpthread -lnuma -lm -mavx

all: cmp_32 msb_32


msb_32: msb_32.c init.c rand.c zipf.c shuffle.c mypapi.c
	${CC} ${CFLAGS} -o msb_32 msb_32.c rand.c init.c zipf.c shuffle.c ${CLIBS} mypapi.c /usr/local/lib/libpapi.a 

cmp_32: cmp_32.c init.c rand.c zipf.c shuffle.c mypapi.c
	${CC} ${CFLAGS} -o cmp_32 cmp_32.c rand.c init.c zipf.c shuffle.c ${CLIBS} mypapi.c /usr/local/lib/libpapi.a 

clean:
	rm -f msb_32 cmp_32
