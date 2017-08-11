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


#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <emmintrin.h>
#include <unistd.h>
#include <sched.h>
#include <numa.h>
#undef _GNU_SOURCE_

#include "rand.h"


static uint64_t micro_time(void)
{
	struct timeval t;
	struct timezone z;
	gettimeofday(&t, &z);
	return t.tv_sec * 1000000 + t.tv_usec;
}

static int hardware_threads(void)
{
	char name[40];
	struct stat st;
	int cpus = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++cpus);
	} while (stat(name, &st) == 0);
	return cpus;
}

static void cpu_bind(int cpu_id)
{
	int cpus = hardware_threads();
	size_t size = CPU_ALLOC_SIZE(cpus);
	cpu_set_t *cpu_set = CPU_ALLOC(cpus);
	assert(cpu_set != NULL);
	CPU_ZERO_S(size, cpu_set);
	CPU_SET_S(cpu_id, size, cpu_set);
	assert(pthread_setaffinity_np(pthread_self(),
	       size, cpu_set) == 0);
	CPU_FREE(cpu_set);
}

static void memory_bind(int numa_id)
{
	char numa_id_str[12];
	struct bitmask *numa_node;
	sprintf(numa_id_str, "%d", numa_id);
	numa_node = numa_parse_nodestring(numa_id_str);
	numa_set_membind(numa_node);
	numa_free_nodemask(numa_node);
}

static void *mamalloc(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

static void schedule_threads(int *cpu, int *numa_node, int threads, int numa)
{
	int max_numa = numa_max_node() + 1;
	int max_threads = hardware_threads();
	int max_threads_per_numa = max_threads / max_numa;
	int t, threads_per_numa = threads / numa;
	assert(threads >= numa && threads % numa == 0);
	if (numa > max_numa ||
	    threads > max_threads ||
	    threads_per_numa > max_threads_per_numa)
		for (t = 0 ; t != threads ; ++t) {
			cpu[t] = t;
			numa_node[t] = t / threads_per_numa;
		}
	else {
		int *thread_numa = malloc(max_threads * sizeof(int));
		for (t = 0 ; t != max_threads ; ++t)
			thread_numa[t] = numa_node_of_cpu(t);
		for (t = 0 ; t != threads ; ++t) {
			int i, n = t % numa;
			for (i = 0 ; i != max_threads ; ++i)
				if (thread_numa[i] == n) break;
			assert(i != max_threads);
			thread_numa[i] = -1;
			cpu[t] = i;
			if (numa_node != NULL)
				numa_node[t] = n;
			assert(numa_node_of_cpu(i) == n);
		}
		free(thread_numa);
	}
}

static uint64_t rand_64(void)
{
	uint64_t r1 = rand() >> 11;
	uint64_t r2 = rand() >> 11;
	uint64_t r3 = rand() >> 10;
	return r1 | (r2 << 21) | (r3 << 42);
}

static uint64_t compact_32(uint32_t *keys, uint64_t size)
{
	if (!size) return 0;
	uint32_t prev_key = keys[0];
	uint64_t p = 0, u = 0;
	do {
		uint32_t key = keys[p++];
		if (key != prev_key) {
			keys[u++] = prev_key;
			prev_key = key;
		}
	} while (p != size);
	keys[u++] = prev_key;
	return u;
}

static uint64_t compact_64(uint64_t *keys, uint64_t size)
{
	if (!size) return 0;
	uint64_t prev_key = keys[0];
	uint64_t p = 0, u = 0;
	do {
		uint64_t key = keys[p++];
		if (key != prev_key) {
			keys[u++] = prev_key;
			prev_key = key;
		}
	} while (p != size);
	keys[u++] = prev_key;
	return u;
}

static int uint64_compare(const void *p1, const void *p2)
{
	uint64_t x = *((uint64_t*) p1);
	uint64_t y = *((uint64_t*) p2);
	return x < y ? -1 : x > y ? 1 : 0;
}

static uint64_t *generate_unique(uint64_t size, int max_bits)
{
	uint64_t i, extra_size = size;
	assert(max_bits <= 64 && size <= ((uint64_t) 1) << max_bits);
	uint64_t mask = ~0;
	if (max_bits != 64)
		mask = (((uint64_t) 1) << max_bits) - 1;
	uint64_t *data = NULL;
	do {
		extra_size *= 1.1;
		data = realloc(data, extra_size * sizeof(uint64_t));
		for (i = 0 ; i != size ; ++i)
			data[i] = rand_64() & mask;
		qsort(data, extra_size, sizeof(uint64_t), uint64_compare);
	} while (compact_64(data, extra_size) < size);
	return data;
}

typedef struct {
	void **data;
	uint64_t *size;
	uint64_t *cap;
	uint64_t *hh_values;
	int length;
	int threads;
	int numa;
	int max_threads;
	int max_numa;
	int interleaved;
	int bits;
	int hh_bits;
	double hh_percentage;
	int *cpu;
	int *numa_node;
	pthread_barrier_t *barrier;
} init_global_data_t;

typedef struct {
	int id;
	int seed;
	uint64_t checksum;
	init_global_data_t *global;
} init_local_data_t;

static void *init_thread(void *arg)
{
	init_local_data_t *a = (init_local_data_t*) arg;
	init_global_data_t *d = a->global;
	int i, n, id = a->id;
	int numa = d->numa;
	int numa_node = d->numa_node[id];
	int threads = d->threads;
	int threads_per_numa = threads / numa;
	// id in local numa threads
	int numa_local_id = 0;
	for (i = 0 ; i != id ; ++i)
		if (d->numa_node[i] == numa_node)
			numa_local_id++;
	// bind thread and its allocation
	if (threads <= d->max_threads)
		cpu_bind(d->cpu[id]);
	if (numa <= d->max_numa)
		memory_bind(d->numa_node[id]);
	// allocate space
	uint64_t unit = d->length >> 3;
	if (numa_local_id == 0) {
		if (d->interleaved)
			d->data[numa_node] = numa_alloc_interleaved(d->cap[numa_node] * unit);
		else
			d->data[numa_node] = mamalloc(d->cap[numa_node] * unit);
		assert(d->data[numa_node] != NULL);
	}
	pthread_barrier_wait(d->barrier);
	// write over space
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = numa_size / threads_per_numa;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - offset;
	if (!size) pthread_exit(NULL);
	uint64_t prev_offset = offset;
	for (n = 0 ; n != numa_node ; ++n)
		prev_offset += d->size[n];
	uint64_t p, q, checksum = 0;
	rand64_t *gen = rand64_init(a->seed);
	uint64_t mask = (1ull << d->bits) - 1;
	uint64_t hh_mask = (1ull << d->hh_bits) - 1;
	if (d->bits == 64) mask = ~0;
	uint64_t hh_limit = ~0;
	hh_limit *= d->hh_percentage;
	uint64_t *hh = d->hh_values;
	assert(d->data[numa_node] != NULL);
	if (numa_local_id == 0)
		assert((63 & (uint64_t) d->data[numa_node]) == 0);
	if (d->length == 32) {
		uint32_t *data = (uint32_t*) d->data[numa_node];
		data = &data[offset];
		p = q = 0;
		if (d->bits == 0)
			do {
				_mm_stream_si32(&data[p++], 0);
			} while (p != size);
		else if (d->bits < 0)
			do {
				uint64_t x = prev_offset++;
				checksum += x;
				_mm_stream_si32(&data[p++], x);
			} while (p != size);
		else if (d->hh_percentage == 0.0)
			while (p != size) {
				uint64_t x = rand64_next(gen) & mask;
				checksum += x;
				_mm_stream_si32(&data[p++], x);
			}
		else
			while (p != size) {
				uint64_t x;
				if (rand64_next(gen) < hh_limit)
					x = hh[rand64_next(gen) & hh_mask];
				else
					x = rand64_next(gen) & mask;
				checksum += x;
				_mm_stream_si32(&data[p++], x);
			}
	} else if (d->length == 64) {
		uint64_t *data = (uint64_t*) d->data[numa_node];
		data = &data[offset];
		p = q = 0;
		if (d->bits == 0)
			do {
				_mm_stream_si64((long long int*) &data[p++], 0);
			} while (p != size);
		else if (d->bits < 0)
			 do {
				uint64_t x = prev_offset++;
				checksum += x;
				_mm_stream_si64((long long int*) &data[p++], x);
			} while (p != size);
		else if (d->hh_percentage == 0.0)
			while (p != size) {
				uint64_t x = rand64_next(gen) & mask;
				checksum += x;
				_mm_stream_si64((long long int*) &data[p++], x);
			}
		else
			while (p != size) {
				uint64_t x;
				if (rand64_next(gen) < hh_limit)
					x = hh[rand64_next(gen) & hh_mask];
				else
					x = rand64_next(gen) & mask;
				checksum += x;
				_mm_stream_si64((long long int*) &data[p++], x);
			}
	} else abort();
	numa_size = d->cap[numa_node] - d->size[numa_node];
	size = numa_size / threads_per_numa;
	offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - offset;
	if (d->length == 32) {
		uint32_t *data = (uint32_t*) d->data[numa_node];
		data = &data[offset + d->size[numa_node]];
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si32(&data[p], 0);
	} else if (d->length == 64) {
		uint64_t *data = (uint64_t*) d->data[numa_node];
		data = &data[offset + d->size[numa_node]];
		for (p = 0 ; p != size ; ++p)
			_mm_stream_si64((long long int*) &data[p], 0);
	} else abort();
	a->checksum = checksum;
	free(gen);
	pthread_exit(NULL);
}

static uint64_t init(void **data, int length, uint64_t *size, uint64_t *cap, int threads, int numa,
                     int bits, double hh_percentage, int hh_bits, int interleaved)
{	int t;
	assert(length == 32 || length == 64);
	assert(size != NULL || cap != NULL);
	if (size == NULL) size = cap;
	if (cap == NULL) cap = size;
	assert(bits >= -1 && bits <= length);
	assert(bits <= 0 || hh_bits < bits);
	assert(hh_bits >= 0 && hh_bits <= length);
	assert(hh_percentage >= 0.0 && hh_percentage <= 1.0);
	assert(hh_percentage == 0.0 || (hh_bits >= 0 && hh_bits <= 20));
	if (bits <= 0 || hh_percentage == 0.0) hh_bits = 0;
	pthread_barrier_t barrier;
	init_global_data_t global;
	global.hh_values = generate_unique(1 << hh_bits, bits);
	global.threads = threads;
	global.numa = numa;
	global.max_threads = hardware_threads();
	global.max_numa = numa_max_node() + 1;
	global.length = length;
	global.interleaved = interleaved;
	global.hh_percentage = hh_percentage;
	global.data = data;
	global.size = size;
	global.cap = cap;
	global.bits = bits;
	global.hh_bits = hh_bits;
	global.barrier = &barrier;
	global.cpu = malloc(global.threads * sizeof(int));
	global.numa_node = malloc(global.threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	pthread_barrier_init(&barrier, NULL, threads);
	init_local_data_t *thread_data;
	thread_data = malloc(threads * sizeof(init_local_data_t));
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	for (t = 0 ; t != threads ; ++t) {
		thread_data[t].id = t;
		thread_data[t].seed = rand();
		thread_data[t].global = &global;
		pthread_create(&id[t], NULL, init_thread, (void*) &thread_data[t]);
	}
	for (t = 0 ; t != threads ; ++t)
		pthread_join(id[t], NULL);
	uint64_t checksum = 0;
	for (t = 0 ; t != threads ; ++t)
		checksum += thread_data[t].checksum;
	pthread_barrier_destroy(&barrier);
	free(global.numa_node);
	free(global.cpu);
	free(thread_data);
	free(id);
	return checksum;
}

uint64_t init_32(uint32_t **data, uint64_t *size, uint64_t *cap, int threads, int numa,
                 int bits, double hh_percentage, int hh_bits, int interleaved)
{
	return init((void*) data, 32, size, cap, threads, numa, bits, hh_percentage, hh_bits, interleaved);
}

uint64_t init_64(uint64_t **data, uint64_t *size, uint64_t *cap, int threads, int numa,
                 int bits, double hh_percentage, int hh_bits, int interleaved)
{
	return init((void*) data, 64, size, cap, threads, numa, bits, hh_percentage, hh_bits, interleaved);
}
