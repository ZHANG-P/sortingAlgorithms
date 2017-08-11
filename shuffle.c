/* Copyright (c) 2013
 * The Trustees of Columbia University in the City of New York
 * All rights reserved.
 *
 * Author:  Orestis Polychroniou  (orestis@cs.columbia.edu)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <numa.h>

#include "rand.h"


static int hardware_threads(void)
{
	char cpu_name[40];
	struct stat st;
	strcpy(cpu_name, "/sys/devices/system/cpu/cpu");
	char *cpu_num_ptr = &cpu_name[strlen(cpu_name)];
	int cpu_num = -1;
	do {
		sprintf(cpu_num_ptr, "%d", ++cpu_num);
	} while (stat(cpu_name, &st) == 0);
	return cpu_num;
}

static uint64_t micro_time(void)
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec * 1000000 + t.tv_usec;
}

static void *mamalloc(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

static void randomize_bytes(uint8_t *bytes, uint64_t size, rand64_t *gen,
			    int thread_id, int threads)
{
	// set sub-array for current thread
	uint64_t i, local_size = (size / threads) & ~15;
	uint8_t *local_bytes = &bytes[local_size * thread_id];
	if (thread_id + 1 == threads)
		local_size = size - local_size * thread_id;
	uint64_t local_size_div_8 = local_size >> 3;
	uint64_t *local_bytes_64 = (uint64_t*) local_bytes;
	// generate random bytes
	for (i = 0 ; i != local_size_div_8 ; ++i)
		local_bytes_64[i] = rand64_next(gen);
	uint64_t j = rand64_next(gen);
	for (i <<= 3 ; i != local_size ; ++i) {
		local_bytes[i] = j;
		j >>= 8;
	}
}

static uint64_t xadd(volatile uint64_t *p, uint64_t v)
{
	return __sync_fetch_and_add(p, v);
}

static inline uint64_t mulhi(uint64_t x, uint64_t y)
{
	uint64_t l, h;
	asm("mulq	%2"
	: "=a"(l), "=d"(h)
	: "r"(y), "a"(x)
	: "cc");
	return h;
}

static void local_shuffle_32(uint32_t *data, uint8_t *random_bytes,
			     uint64_t size, rand64_t *gen)
{
	uint64_t i, j, p;
	if (size <= 40000) {
		for (i = 0 ; i != size ; ++i) {
			j = mulhi(rand64_next(gen), size - i) + i;
			uint32_t temp = data[i];
			data[i] = data[j];
			data[j] = temp;
		}
		return;
	}
	// histogram
	uint64_t count[256];
	for (p = 0 ; p != 256 ; ++p)
		count[p] = 0;
	// fill random bytes
	p = rand64_next(gen);
	for (i = j = 0 ; i != size ; ++i) {
		random_bytes[i] = p;
		count[p & 255]++;
		p >>= 8;
		if ((++j & 7) == 0)
			p = rand64_next(gen);
	}
	// offsets
	uint64_t offsets[256];
	for (i = p = 0 ; p != 256 ; ++p) {
		i += count[p];
		offsets[p] = i;
	}
	assert(i == size);
	// partition in place
	for (j = p = 0 ; p != 256 ; ++p)
		if (count[p]) break;
	while (p != 256) {
		uint32_t item = data[i = j];
		do {
			i = --offsets[random_bytes[i]];
			uint32_t temp = data[i];
			data[i] = item;
			item = temp;
		} while (i != j);
		do {
			j += count[p++];
		} while (p != 256 && j == offsets[p]);
	}
	// call recursive for sub-parts
	for (i = p = 0 ; p != 256 ; ++p) {
		local_shuffle_32(&data[i], &random_bytes[i], count[p], gen);
		i += count[p];
	}
}

static void local_shuffle_64(uint64_t *data, uint8_t *random_bytes,
			     size_t size, rand64_t *gen)
{
	uint64_t i, j, p;
	if (size <= 20000) {
		for (i = 0 ; i != size ; ++i) {
			j = mulhi(rand64_next(gen), size - i) + i;
			uint64_t temp = data[i];
			data[i] = data[j];
			data[j] = temp;
		}
		return;
	}
	// histogram
	uint64_t count[256];
	for (p = 0 ; p != 256 ; ++p)
		count[p] = 0;
	// fill random bytes
	p = rand64_next(gen);
	for (i = j = 0 ; i != size ; ++i) {
		random_bytes[i] = p;
		count[p & 255]++;
		p >>= 8;
		if ((++j & 7) == 0)
			p = rand64_next(gen);
	}
	// offsets
	uint64_t offsets[256];
	for (i = p = 0 ; p != 256 ; ++p) {
		i += count[p];
		offsets[p] = i;
	}
	assert(i == size);
	// partition in place
	for (j = p = 0 ; p != 256 ; ++p)
		if (count[p]) break;
	while (p != 256) {
		uint64_t item = data[i = j];
		do {
			i = --offsets[random_bytes[i]];
			uint64_t temp = data[i];
			data[i] = item;
			item = temp;
		} while (i != j);
		do {
			j += count[p++];
		} while (p != 256 && j == offsets[p]);
	}
	// call recursive for sub-parts
	for (i = p = 0 ; p != 256 ; ++p) {
		local_shuffle_64(&data[i], &random_bytes[i], count[p], gen);
		i += count[p];
	}
}

static uint64_t *known_partition_32(uint32_t *keys, uint32_t *keys_out,
				    uint8_t *parts, uint64_t size, uint64_t **hist,
				    int thread_id, int threads, pthread_barrier_t *barrier)
{
	// inputs and outputs must be aligned
	assert(0 == (15 & (size_t) keys));
	assert(0 == (15 & (size_t) keys_out));
	// set sub-array for current thread
	uint64_t local_size = (size / threads) & ~15;
	uint32_t *local_keys = &keys[local_size * thread_id];
	uint8_t *local_parts = &parts[local_size * thread_id];
	if (thread_id + 1 == threads)
		local_size = size - local_size * thread_id;
	assert((local_size & 3) == 0);
	// initialize histogram
	uint64_t i, j, p; int t;
	uint64_t partitions = 256;
	uint64_t *local_hist = hist[thread_id];
	for (p = 0 ; p != partitions ; ++p)
		local_hist[p] = 0;
	// main histogram loop
	__m128i h, z = _mm_setzero_si128();
	for (i = p = 0 ; i != local_size ; i += 4) {
		uint32_t *l = (uint32_t*) &local_parts[i];
		asm("movd	(%1), %0" : "=x"(h) : "r"(l));
		h = _mm_unpacklo_epi8(h, z);
		h = _mm_unpacklo_epi16(h, z);
		for (j = 0 ; j != 4 ; ++j) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			local_hist[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete histogram generation
	pthread_barrier_wait(&barrier[0]);
	// initialize buffer
	uint64_t *index = malloc(partitions * sizeof(uint64_t));
	uint32_t *buf = mamalloc((partitions << 4) * sizeof(uint32_t));
	uint64_t *buf_64 = (uint64_t*) buf;
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t != thread_id ; ++t)
			i += hist[t][p];
		index[p] = i;
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	// main partitioning loop
	for (i = p = 0 ; i != local_size ; i += 4) {
		uint32_t *l = (uint32_t*) &local_parts[i];
		asm("movd	(%1), %0" : "=x"(h) : "r"(l));
		__m128i k = _mm_load_si128((__m128i*) &local_keys[i]);
		h = _mm_unpacklo_epi8(h, z);
		h = _mm_unpacklo_epi16(h, z);
		for (j = 0 ; j != 4 ; ++j) {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src_64 = &buf_64[p << 3];
			uint32_t *src = (uint32_t*) src_64;
			uint64_t indexl = index[p]++;
			uint64_t offset = indexl & 15;
			asm("movd	%0, (%1,%2,4)" :: "x"(k), "r"(src), "r"(offset) : "memory");
			if (offset == 15) {
				uint32_t *dst = &keys_out[indexl - 15];
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[8]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[12]);
				_mm_stream_si128((__m128i*) &dst[0], r0);
				_mm_stream_si128((__m128i*) &dst[4], r1);
				_mm_stream_si128((__m128i*) &dst[8], r2);
				_mm_stream_si128((__m128i*) &dst[12],r3);
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete main partition part
	pthread_barrier_wait(&barrier[1]);
	// flush remaining items from buffers to output
	for (p = 0 ; p != partitions ; ++p) {
		uint32_t *src = &buf[p << 4];
		uint64_t indexl = index[p];
		uint64_t remain = indexl & 15;
		uint64_t offset = 0;
		if (remain > local_hist[p])
			offset = remain - local_hist[p];
		indexl -= remain - offset;
		while (offset != remain)
			_mm_stream_si32(&keys_out[indexl++], src[offset++]);
	}
	// wait all threads to complete last partition part
	pthread_barrier_wait(&barrier[2]);
	// check sizes of partitions and free buffer
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t <= thread_id ; ++t)
			i += hist[t][p];
		assert(index[p] == i);
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	free(index);
	free(buf);
	// compute total histogram
	uint64_t *result_hist = calloc(partitions, sizeof(uint64_t));
	for (t = 0 ; t != threads ; ++t)
		for (p = 0 ; p != partitions ; ++p)
			result_hist[p] += hist[t][p];
	return result_hist;
}

static uint64_t *known_partition_64(uint64_t *keys, uint64_t *keys_out,
				    uint8_t *parts, uint64_t size, uint64_t **hist,
				    int thread_id, int threads, pthread_barrier_t *barrier)
{
	// inputs and outputs must be aligned
	assert(0 == (15 & (size_t) keys));
	assert(0 == (63 & (size_t) keys_out));
	// set sub-array for current thread
	uint64_t local_size = (size / threads) & ~15;
	uint64_t *local_keys = &keys[local_size * thread_id];
	uint8_t *local_parts = &parts[local_size * thread_id];
	if (thread_id + 1 == threads)
		local_size = size - local_size * thread_id;
	assert((local_size & 3) == 0);
	// initialize histogram
	uint64_t i, j, p; int t;
	uint64_t partitions = 256;
	uint64_t *local_hist = hist[thread_id];
	for (p = 0 ; p != partitions ; ++p)
		local_hist[p] = 0;
	// main histogram loop
	__m128i h, z = _mm_setzero_si128();
	for (i = p = 0 ; i != local_size ; i += 4) {
		uint32_t *l = (uint32_t*) &local_parts[i];
		asm("movd	(%1), %0" : "=x"(h) : "r"(l));
		h = _mm_unpacklo_epi8(h, z);
		h = _mm_unpacklo_epi16(h, z);
		for (j = 0 ; j != 4 ; ++j) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			local_hist[p]++;
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete histogram generation
	pthread_barrier_wait(&barrier[0]);
	// initialize buffer
	uint64_t *buf = mamalloc((partitions << 3) * sizeof(uint64_t));
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t != thread_id ; ++t)
			i += hist[t][p];
		buf[(p << 3) | 7] = i;
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	// main partitioning loop
	for (i = p = 0 ; i != local_size ; i += 4) {
		uint32_t *l = (uint32_t*) &local_parts[i];
		asm("movd	(%1), %0" : "=x"(h) : "r"(l));
		__m128i k12 = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i k34 = _mm_load_si128((__m128i*) &local_keys[i + 2]);
		k12 = _mm_shuffle_epi32(k12, _MM_SHUFFLE(3, 1, 2, 0));
		k34 = _mm_shuffle_epi32(k34, _MM_SHUFFLE(3, 1, 2, 0));
		h = _mm_unpacklo_epi8(h, z);
		h = _mm_unpacklo_epi16(h, z);
		__m128i k_L = _mm_unpacklo_epi64(k12, k34);
		__m128i k_H = _mm_unpackhi_epi64(k12, k34);
		h = _mm_slli_epi32(h, 3);
		for (j = 0 ; j != 4 ; ++j) {
			// extract partition
			asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
			// offset in the cache line pair
			uint64_t *src = &buf[p];
			uint64_t index = src[7]++;
			uint64_t offset = index & 7;
			__m128i k = _mm_unpacklo_epi32(k_L, k_H);
			_mm_storel_epi64((__m128i*) &src[offset], k);
			if (offset == 7) {
				uint64_t *dst = &keys_out[index - 7];
				__m128i r0 = _mm_load_si128((__m128i*) &src[0]);
				__m128i r1 = _mm_load_si128((__m128i*) &src[2]);
				__m128i r2 = _mm_load_si128((__m128i*) &src[4]);
				__m128i r3 = _mm_load_si128((__m128i*) &src[6]);
				_mm_stream_si128((__m128i*) &dst[0], r0);
				_mm_stream_si128((__m128i*) &dst[2], r1);
				_mm_stream_si128((__m128i*) &dst[4], r2);
				_mm_stream_si128((__m128i*) &dst[6], r3);
				src[7] = index + 1;
			}
			// rotate
			h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
			k_L = _mm_shuffle_epi32(k_L, _MM_SHUFFLE(0, 3, 2, 1));
			k_H = _mm_shuffle_epi32(k_H, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	// wait all threads to complete main partition part
	pthread_barrier_wait(&barrier[1]);
	// flush remaining items from buffers to output
	for (p = 0 ; p != partitions ; ++p) {
		uint64_t *src = &buf[p << 3];
		uint64_t index = src[7];
		uint64_t remain = index & 7;
		uint64_t offset = 0;
		if (remain > local_hist[p])
			offset = remain - local_hist[p];
		index -= remain - offset;
		while (offset != remain)
			_mm_stream_si64((long long int*) &keys_out[index++], src[offset++]);
	}
	// wait all threads to complete last partition part
	pthread_barrier_wait(&barrier[2]);
	// check sizes of partitions and free buffer
	for (i = p = 0 ; p != partitions ; ++p) {
		for (t = 0 ; t <= thread_id ; ++t)
			i += hist[t][p];
		assert(buf[(p << 3) | 7] == i);
		for (; t != threads ; ++t)
			i += hist[t][p];
	}
	assert(i == size);
	free(buf);
	// compute total histogram
	uint64_t *result_hist = calloc(partitions, sizeof(uint64_t));
	for (t = 0 ; t != threads ; ++t)
		for (p = 0 ; p != partitions ; ++p)
			result_hist[p] += hist[t][p];
	return result_hist;
}

typedef struct {
	int mode;
	int seed;
	int threads;
	int thread_id;
	uint64_t size;
	void *data;
	void *space;
	uint8_t *random_bytes;
	uint64_t **parallel_hist;
	pthread_barrier_t *barrier;
	volatile uint64_t *part_counter;
} thread_data_t;

static void *run(void *arg)
{
	thread_data_t *d = (thread_data_t*) arg;
	uint64_t p_target, p, unique, *hist;
	uint64_t partitions = 256;
	assert(d->mode == 32 || d->mode == 64);
	uint32_t *data_32 = (uint32_t*) d->data;
	uint64_t *data_64 = (uint64_t*) d->data;
	uint32_t *space_32 = (uint32_t*) d->space;
	uint64_t *space_64 = (uint64_t*) d->space;
	rand64_t *gen = rand64_init(d->seed);
	randomize_bytes(d->random_bytes, d->size, gen,
			d->thread_id, d->threads);
	if (d->mode == 32)
		hist = known_partition_32(data_32, space_32, d->random_bytes,
					  d->size, d->parallel_hist,
					  d->thread_id, d->threads, d->barrier);
	else
		hist = known_partition_64(data_64, space_64, d->random_bytes,
					  d->size, d->parallel_hist,
					  d->thread_id, d->threads, d->barrier);
	while ((p_target = xadd(d->part_counter, 1)) < partitions) {
		uint64_t offset = 0;
		for (p = 0 ; p != p_target ; ++p)
			offset += hist[p];
		uint64_t size = hist[p];
		if (d->mode == 32) {
			local_shuffle_32(&space_32[offset], &d->random_bytes[offset], size, gen);
			memcpy(&data_32[offset], &space_32[offset], size * sizeof(uint32_t));
		} else {
			local_shuffle_64(&space_64[offset], &d->random_bytes[offset], size, gen);
			memcpy(&data_64[offset], &space_64[offset], size * sizeof(uint64_t));
		}
	}
	free(hist);
	pthread_exit(NULL);
}

static void shuffle(void *array, uint64_t size, int mode)
{
	assert(mode == 32 || mode == 64);
	uint64_t length = mode >> 2;
	int t, threads = hardware_threads();
	if (threads > 256) threads = 256;
	pthread_t id[threads];
	thread_data_t data[threads];
	uint64_t *parallel_hist[threads];
	pthread_barrier_t barrier[3];
	volatile uint64_t part_counter = 0;
	uint8_t *random_bytes = numa_alloc_interleaved(size);
	assert(random_bytes != NULL);
	void *space = numa_alloc_interleaved(size * length);
	assert(space != NULL);
	for (t = 0 ; t != threads ; ++t)
		parallel_hist[t] = malloc(256 * sizeof(uint64_t));
	for (t = 0 ; t != 3 ; ++t)
		pthread_barrier_init(&barrier[t], NULL, threads);
	for (t = 0 ; t != threads ; ++t) {
		data[t].mode = mode;
		data[t].thread_id = t;
		data[t].threads = threads;
		data[t].seed = rand();
		data[t].data = array;
		data[t].size = size;
		data[t].space = space;
		data[t].random_bytes = random_bytes;
		data[t].parallel_hist = parallel_hist;
		data[t].barrier = barrier;
		data[t].part_counter = &part_counter;
		pthread_create(&id[t], NULL, run, (void*) &data[t]);
	}
	for (t = 0 ; t != threads ; ++t)
		pthread_join(id[t], NULL);
	for (t = 0 ; t != 3 ; ++t)
		pthread_barrier_destroy(&barrier[t]);
	for (t = 0 ; t != threads ; ++t)
		free(parallel_hist[t]);
	numa_free(space, size * length);
}

void shuffle_32(uint32_t *array, uint64_t size)
{
	assert((size & 3) == 0);
	shuffle((void*) array, size, 32);
}

void shuffle_64(uint64_t *array, uint64_t size)
{
	assert((size & 3) == 0);
	shuffle((void*) array, size, 64);
}

/*
int main(int argc, char **argv)
{
	int mil = argc > 1 ? atoi(argv[1]) : 1000;
	assert(mil >= 0 && mil <= 20000);
	uint64_t i, size = mil * 1000 * 1000;
	uint32_t *keys_32 = numa_alloc_interleaved(size * sizeof(uint32_t));
	for (i = 0 ; i != size ; ++i)
		keys_32[i] = i;
	uint64_t time_32 = micro_time();
	shuffle_32(keys_32, size);
	time_32 = micro_time() - time_32;
	numa_free(keys_32, size * sizeof(uint32_t));
	uint64_t *keys_64 = numa_alloc_interleaved(size * sizeof(uint64_t));
	for (i = 0 ; i != size ; ++i)
		keys_64[i] = i;
	uint64_t time_64 = micro_time();
	shuffle_64(keys_64, size);
	time_64 = micro_time() - time_64;
	numa_free(keys_64, size * sizeof(uint64_t));
	double sec_32 = time_32 / 1000000.0;
	double sec_64 = time_64 / 1000000.0;
	fprintf(stderr, "Shuffled %d million 32-bit keys in %.2f sec\n", mil, sec_32);
	fprintf(stderr, "Shuffled %d million 64-bit keys in %.2f sec\n", mil, sec_64);
	return EXIT_SUCCESS;
}
*/
