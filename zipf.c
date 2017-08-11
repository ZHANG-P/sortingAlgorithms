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
#include <math.h>
#include <numa.h>
#include <errno.h>
#undef _GNU_SOURCE

#include "rand.h"


static uint64_t micro_time(void)
{
	struct timeval t;
	struct timezone z;
	gettimeofday(&t, &z);
	return t.tv_sec * 1000000 + t.tv_usec;
}

static int cpus(void)
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

static void cpu_bind(int cpu_id)
{
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
}

static void memory_bind(int cpu_id)
{
	char numa_id_str[12];
	struct bitmask *numa_node;
	int numa_id = numa_node_of_cpu(cpu_id);
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
	int t, max_threads = cpus();
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

typedef struct {
	uint32_t **data;
	uint64_t *size;
	double *prob;
	uint32_t *values;
	double theta;
	double *prob_factor;
	int threads;
	int numa;
	int *cpu;
	int *numa_node;
	double *factors;
	pthread_barrier_t *barrier;
} global_data_t;

typedef struct {
	int id;
	uint32_t seed;
	uint64_t checksum;
	uint64_t *bits;
	global_data_t *global;
} thread_data_t;

static void *zipf_thread(void *arg)
{
	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	int i, t, id = a->id;
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
	cpu_bind(d->cpu[id]);
	memory_bind(d->cpu[id]);
	// allocate space
	uint64_t numa_size = d->size[numa_node];
	if (numa_local_id == 0 && d->data[numa_node] == NULL)
		d->data[numa_node] = mamalloc(numa_size * sizeof(uint32_t));
	// offsets of probability array
	uint64_t prob_max = ((uint64_t) 1) << 32;
	uint64_t prob_size = prob_max / threads;
	uint64_t prob_offset = prob_size * id;
	if (id + 1 == threads)
		prob_size = prob_max - prob_offset;
	uint64_t prob_end = prob_offset + prob_size;
	// compute probability sum (across all threads)
	double prob_factor = 0.0;
	double neg_theta = 0.0 - d->theta;
	uint64_t p;
	for (p = prob_offset ; p != prob_end ; ++p)
		prob_factor += pow(p + 1.0, neg_theta);
	d->prob_factor[id] = prob_factor;
	// synchronize
	pthread_barrier_wait(&d->barrier[0]);
	// compute total factor from all threads
	prob_factor = 0.0;
	for (t = 0 ; t != threads ; ++t)
		prob_factor += d->prob_factor[t];
	prob_factor = 1.0 / prob_factor;
	// compute probability sum up to this point
	double prob_sum = 0.0;
	for (t = 0 ; t != id ; ++t)
		prob_sum += d->prob_factor[t];
	prob_sum *= prob_factor;
	// set probability array
	double *prob = d->prob;
	for (p = prob_offset ; p != prob_end ; ++p) {
		prob_sum += prob_factor * pow(p + 1.0, neg_theta);
		prob[p] = prob_sum;
	}
	// synchronize
	pthread_barrier_wait(&d->barrier[1]);
	// offsets of rank array
	uint64_t size = numa_size / threads_per_numa;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - offset;
	uint32_t *data = &d->data[numa_node][offset];
	uint32_t *data_end = &data[size];
	uint32_t *data_start = data;
	// generate zipfian ranks
	rand64_t *gen = rand64_init(a->seed);
	uint64_t tim = micro_time();
	double max_inv = 1.0 / ((uint64_t) ~0);
	while (data != data_end) {
		uint64_t low = 0;
		uint64_t high = prob_max - 1;
		double x = rand64_next(gen) * max_inv;
		do {
			uint64_t mid = (low + high) >> 1;
			if (x > prob[mid])
				low = mid + 1;
			else
				high = mid;
		} while (low != high);
		_mm_stream_si32(data++, low);
	}
	// generate zipfian ranks
	tim = micro_time() - tim;
	free(gen);
	// synchronize (wait for 32-bit values to be filled)
	pthread_barrier_wait(&d->barrier[2]);
	// swap with random 32-bit values
	uint64_t *bits = calloc(33, sizeof(uint64_t));
	uint32_t *values = d->values;
	uint64_t checksum = 0;
	data = data_start;
	while (data != data_end) {
		uint32_t rank = *data;
		uint32_t value = values[rank];
		// update checksum
		checksum += value;
		// update rank counting
		int32_t log_rank;
		asm("bsrl	%1, %0\n\t"
		    "cmovzl	%2, %0"
		 : "=r"(log_rank) : "r"(rank), "r"(-1) : "cc");
		bits[log_rank + 1]++;
		// write value
		_mm_stream_si32(data++, value);
	}
	a->bits = bits;
	a->checksum = checksum;
	pthread_exit(NULL);
}

uint64_t zipf_32(uint32_t **data, uint64_t *size, uint32_t *values,
		 int numa, double theta, uint64_t bits[33])
{
	int max_threads = cpus();
	int i, t, threads = 0;
	uint64_t o = 1;
	for (t = 0 ; t != max_threads ; ++t)
		if (numa_node_of_cpu(t) < numa)
			threads++;
	pthread_barrier_t barrier[3];
	pthread_barrier_init(&barrier[0], NULL, threads);
	pthread_barrier_init(&barrier[1], NULL, threads);
	pthread_barrier_init(&barrier[2], NULL, threads);
	global_data_t global;
	global.values = values;
	global.prob = numa_alloc_interleaved((o << 32) * sizeof(double));
	global.threads = threads;
	global.theta = theta;
	global.numa = numa;
	global.data = data;
	global.size = size;
	global.prob_factor = malloc(threads * sizeof(double));
	global.barrier = barrier;
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	thread_data_t *thread_data = malloc(threads * sizeof(thread_data_t));
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	for (t = 0 ; t != threads ; ++t) {
		thread_data[t].id = t;
		thread_data[t].seed = rand();
		thread_data[t].global = &global;
		pthread_create(&id[t], NULL, zipf_thread, (void*) &thread_data[t]);
	}
	uint64_t checksum = 0;
	for (t = 0 ; t != threads ; ++t) {
		pthread_join(id[t], NULL);
		checksum += thread_data[t].checksum;
	}
	if (bits != NULL) {
		for (i = 0 ; i != 33 ; ++i)
			bits[i] = 0;
		for (t = 0 ; t != threads ; ++t)
			for (i = 0 ; i != 33 ; ++i)
				bits[i] += thread_data[t].bits[i];
	}
	for (t = 0 ; t != threads ; ++t)
		free(thread_data[t].bits);
	for (t = 0 ; t != 2 ; ++t)
		pthread_barrier_destroy(&barrier[t]);
	numa_free(global.prob, (o << 32) * sizeof(uint32_t));
	numa_free(global.values, (o << 32) * sizeof(uint32_t));
	free(global.prob_factor);
	free(global.numa_node);
	free(thread_data);
	free(global.cpu);
	free(id);
	return checksum;
}

uint64_t zipf_64(uint64_t **data, uint64_t *size, uint64_t values,
		 int numa, double theta, uint64_t bits[65])
{
	fprintf(stderr, "64-bit Zipf not supported!\n");
	abort();
	return 0;
}

/*
int main(int argc, char **argv)
{
	int i, numa = numa_max_node() + 1;
	uint64_t tuples = argc > 1 ? atoi(argv[1]) : 1000;
	double theta = argc > 2 ? atof(argv[2]) : 1.0;
	char *output = argc > 3 ? argv[3] : NULL;
	tuples *= 1000000;
	uint64_t bits[33];
	uint32_t *data[numa];
	uint64_t size[numa];
	FILE *fp = NULL;
	if (output != NULL) {
		fp = fopen(output, "w");
		assert(fp != NULL);
	}
	for (i = 0 ; i != numa ; ++i) {
		size[i] = tuples / numa;
		data[i] = NULL;
	}
	fprintf(stderr, "Theta: %.2f\n", theta);
	uint64_t t = micro_time();
	zipf(data, size, numa, theta, bits);
	t = micro_time() - t;
	uint64_t s = 0;
	for (i = 0 ; i != 33 ; ++i) {
		s += bits[i];
		fprintf(stderr, "%2d bits: %.2f%% (%ld/%ld)\n",
			i, s * 100.0 / tuples, s, tuples);
	}
	fprintf(stderr, "Time: %ld us\n", t);
	if (fp != NULL) {
		for (i = 0 ; i != numa ; ++i) {
			uint32_t *d = data[i];
			uint64_t rem = size[i];
			while (rem) {
				uint64_t size = rem;
				if (size > 1024) size = 1024;
				size = fwrite(d, 4, size, fp);
				assert(size > 0);
				rem -= size;
				d += size;
			}
		}
		fclose(fp);
		fprintf(stderr, "Written\n");
	}
	return EXIT_SUCCESS;
}
*/
