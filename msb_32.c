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
#include <sched.h>
#include <numa.h>
#undef _GNU_SOURCE

#ifdef AVX
#include <immintrin.h>
#else
#include <smmintrin.h>
#endif

#include "rand.h"
#include "util.h"
#include "papi.h"

uint64_t micro_time(void)
{
	struct timeval t;
	struct timezone z;
	gettimeofday(&t, &z);
	return t.tv_sec * 1000000 + t.tv_usec;
}

int hardware_threads(void)
{
	char name[40];
	struct stat st;
	int cpus = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++cpus);
	} while (stat(name, &st) == 0);
	return cpus;
}

void cpu_bind(int cpu_id)
{
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
}

void memory_bind(int cpu_id)
{
	char numa_id_str[12];
	struct bitmask *numa_node;
	int numa_id = numa_node_of_cpu(cpu_id);
	sprintf(numa_id_str, "%d", numa_id);
	numa_node = numa_parse_nodestring(numa_id_str);
	numa_set_membind(numa_node);
	numa_free_nodemask(numa_node);
}

void *mamalloc(size_t size)
{
	void *ptr = NULL;
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
}

uint8_t log_2(uint64_t x)
{
	uint8_t p = 0;
	uint64_t o = 1;
	while ((o << p) < x) p++;
	return p;
}

typedef struct {
	uint32_t x[4];
} u4; u4 u4_v;

typedef struct {
	uint16_t x[8];
} u8; u8 u8_v;

typedef struct {
	int32_t x[4];
} i4; i4 i4_v;

typedef struct {
	int16_t x[8];
} i8; i8 i8_v;

inline uint64_t min(uint64_t x, uint64_t y) { return x < y ? x : y; }
inline uint64_t max(uint64_t x, uint64_t y) { return x > y ? x : y; }

int uint32_compare(const void *x, const void *y)
{
	uint32_t a = *((uint32_t*) x);
	uint32_t b = *((uint32_t*) y);
	return a < b ? -1 : (a > b ? 1 : 0);
}

inline uint64_t mulhi(uint64_t x, uint64_t y)
{
	uint64_t l, h;
	asm("mulq	%2"
	: "=a"(l), "=d"(h)
	: "r"(y), "a"(x)
	: "cc");
	return h;
}

inline uint64_t binary_search_32(uint32_t *keys, uint64_t size, uint32_t key)
{
	uint64_t low = 0;
	uint64_t high = size;
	while (low < high) {
		uint64_t mid = (low + high) >> 1;
		if (key > keys[mid])
			low = mid + 1;
		else
			high = mid;
	}
#ifdef BG
	assert(low == 0 || key > keys[low - 1]);
	assert(low == size || key <= keys[low]);
#endif
	return low;
}

inline uint64_t binary_search_64(uint64_t *keys, uint64_t size, uint64_t key)
{
	uint64_t low = 0;
	uint64_t high = size;
	while (low < high) {
		uint64_t mid = (low + high) >> 1;
		if (key > keys[mid])
			low = mid + 1;
		else
			high = mid;
	}
#ifdef BG
	assert(low == 0 || key > keys[low - 1]);
	assert(low == size || key <= keys[low]);
#endif
	return low;
}

void schedule_threads(int *cpu, int *numa_node, int threads, int numa)
{
	int max_numa = numa_max_node() + 1;
	int max_threads = hardware_threads();
	int max_threads_per_numa = max_threads / max_numa;
	int t, threads_per_numa = threads / numa;
	assert(numa > 0 && threads >= numa && threads % numa == 0);
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

void range_histogram(uint32_t *keys, uint8_t *ranges, uint64_t size,
		     uint64_t *count, uint32_t delim[])
{
	assert((15 & (uint64_t) keys) == 0);
	assert((15 & (uint64_t) delim) == 0);
	assert((size & 3) == 0);
	if (!size) return;
	uint32_t *keys_end = &keys[size];
	uint64_t p = 0, q = 0; uint8_t i;
	uint64_t convert = 1 << 31;
	uint32_t *cdelim = mamalloc(128 * sizeof(uint32_t));
	for (p = 0 ; p != 128 ; ++p)
		cdelim[p] = delim[p] - convert;
	__m128i c = _mm_set1_epi32(convert);
	__m128i d1 = _mm_set1_epi32(cdelim[15]);
	__m128i d2 = _mm_set1_epi32(cdelim[31]);
	__m128i d3 = _mm_set1_epi32(cdelim[47]);
	__m128i d4 = _mm_set1_epi32(cdelim[63]);
	__m128i d5 = _mm_set1_epi32(cdelim[79]);
	__m128i d6 = _mm_set1_epi32(cdelim[95]);
	__m128i d7 = _mm_set1_epi32(cdelim[111]);
	__m128i I = _mm_set1_epi32(~0);
	int32_t *ranges_32 = (int32_t*) ranges;
	do {
		__m128i k = _mm_load_si128((__m128i*) keys);
		keys += 4;
		k = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k, d4);
		__m128i d26 = _mm_blendv_epi8(d2, d6, e1);
		__m128i d15 = _mm_blendv_epi8(d1, d5, e1);
		__m128i d37 = _mm_blendv_epi8(d3, d7, e1);
		__m128i e2 = _mm_cmpgt_epi32(k, d26);
		__m128i d1357 = _mm_blendv_epi8(d15, d37, e2);
		__m128i e3 = _mm_cmpgt_epi32(k, d1357);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e3);
		r = _mm_slli_epi32(r, 4);
		uint64_t ra = 0;
		for (i = 0 ; i != 32 ; i += 8) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(r), "0"(p));
			__m128i di1 = _mm_load_si128((__m128i*) &cdelim[p]);
			__m128i di2 = _mm_load_si128((__m128i*) &cdelim[p + 4]);
			__m128i di3 = _mm_load_si128((__m128i*) &cdelim[p + 8]);
			__m128i di4 = _mm_load_si128((__m128i*) &cdelim[p + 12]);
			__m128i k_x4 = _mm_shuffle_epi32(k, 0);
			__m128i f1 = _mm_cmpgt_epi32(k_x4, di1);
			__m128i f2 = _mm_cmpgt_epi32(k_x4, di2);
			__m128i f3 = _mm_cmpgt_epi32(k_x4, di3);
			__m128i f4 = _mm_cmpgt_epi32(k_x4, di4);
			__m128i f12 = _mm_packs_epi32(f1, f2);
			__m128i f34 = _mm_packs_epi32(f3, f4);
			__m128i f = _mm_packs_epi16(f12, f34);
			f = _mm_xor_si128(f, I);
			asm("pmovmskb	%1, %%ebx" : "=b"(q) : "x"(f), "0"(q));
			asm("bsfl	%%ebx, %%ebx" : "=b"(q) : "0"(q) : "cc");
			p |= q;
			count[p]++;
#ifdef BG
			uint32_t key = keys[(i >> 3) - 4];
			assert(p == 0 || key > delim[p - 1]);
			assert(key <= delim[p]);
#endif
			ra |= (p << i);
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			r = _mm_shuffle_epi32(r, _MM_SHUFFLE(0, 3, 2, 1));
		}
		_mm_stream_si32(ranges_32++, ra);
	} while (keys != keys_end);
#ifdef BG
	keys -= size;
	for (p = 0 ; p != size ; ++p) {
		uint32_t key = keys[p];
		q = ranges[p];
		assert(q == 0 || key > delim[q - 1]);
		assert(key <= delim[q]);
	}
#endif
	free(cdelim);
}

void partition_known(uint32_t *keys, uint32_t *rids, uint8_t *ranges, uint64_t size,
		     uint64_t *sizes, uint32_t *keys_out, uint32_t *rids_out,
		     uint64_t partitions)
{
	// allocate buffer
	if (!size) return;
	assert((size & 15) == 0);
	assert((15 & (uint64_t) keys) == 0);
	assert((15 & (uint64_t) rids) == 0);
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	uint64_t *buf = mamalloc((partitions << 4) * sizeof(uint64_t));
	// initialize offsets
	uint64_t i, j, p = 0;
	for (i = 0 ; i != partitions ; ++i) {
		buf[(i << 4) | 15] = p;
		p += sizes[i];
	}
	assert(p == size);
	// loop of data partitioning
	uint32_t *keys_end = &keys[size];
	do {
		__m128i d = _mm_load_si128((__m128i*) ranges);
		for (i = 0 ; i != 16 ; i += 4) {
			__m128i k = _mm_load_si128((__m128i*) &keys[i]);
			__m128i v = _mm_load_si128((__m128i*) &rids[i]);
			__m128i h = _mm_cvtepu8_epi32(d);
			h = _mm_slli_epi32(h, 4);
			for (j = 0 ; j != 4 ; ++j) {
				// extract partition
				asm("movd	%1, %%eax" : "=a"(p) : "x"(h), "0"(p));
				// offset in the cache line pair
				uint64_t *src = &buf[p];
				uint64_t index = src[15]++;
				uint64_t offset = index & 15;
				// pack and store
				__m128i kvxx = _mm_unpacklo_epi32(k, v);
				_mm_storel_epi64((__m128i*) (float*) &src[offset], kvxx);
				if (offset == 15) {
					float *dest_x = (float*) &keys_out[index - 15];
					float *dest_y = (float*) &rids_out[index - 15];
					// load cache line from cache to 8 128-bit registers
					__m128 r0 = _mm_load_ps((float*) &src[0]);
					__m128 r1 = _mm_load_ps((float*) &src[2]);
					__m128 r2 = _mm_load_ps((float*) &src[4]);
					__m128 r3 = _mm_load_ps((float*) &src[6]);
					__m128 r4 = _mm_load_ps((float*) &src[8]);
					__m128 r5 = _mm_load_ps((float*) &src[10]);
					__m128 r6 = _mm_load_ps((float*) &src[12]);
					__m128 r7 = _mm_load_ps((float*) &src[14]);
					// split first column
					__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
					__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
					// stream first column
					_mm_stream_ps(&dest_x[0], x0);
					_mm_stream_ps(&dest_x[4], x1);
					_mm_stream_ps(&dest_x[8], x2);
					_mm_stream_ps(&dest_x[12],x3);
					// split second column
					__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
					__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
					__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
					__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
					// stream second column
					_mm_stream_ps(&dest_y[0], y0);
					_mm_stream_ps(&dest_y[4], y1);
					_mm_stream_ps(&dest_y[8], y2);
					_mm_stream_ps(&dest_y[12],y3);
					// restore overwritten pointer
					src[15] = index + 1;
				}
				// rotate keys and 4 hashes
				h = _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 3, 2, 1));
				k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
				v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
			}
			// rotate 16 hashes
			d = _mm_shuffle_epi32(d, _MM_SHUFFLE(0, 3, 2, 1));
		}
		// update pointers
		keys += 16;
		rids += 16;
		ranges += 16;
	} while (keys != keys_end);
	// flush remaining items from buffers to output
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t *src = &buf[i << 4];
		uint64_t index = src[15];
		uint64_t rem = index & 15;
		uint64_t off = 0;
		if (rem > sizes[i])
			off = rem - sizes[i];
		index -= rem - off;
		while (off != rem) {
			uint64_t pair = src[off++];
			keys_out[index] = (uint32_t) pair;
			rids_out[index++] = (uint32_t) (pair >> 32);
		}
	}
	free(buf);
}

void check_range(uint32_t *keys, uint64_t size, uint64_t range, uint32_t *delim)
{
	uint32_t *keys_end = &keys[size];
	uint32_t min_key = range == 0 ? 0 : delim[range - 1] + 1;
	uint32_t max_key = delim[range];
	uint64_t p;
	for (p = 0 ; p != size ; ++p) {
		uint32_t key = keys[p];
		assert(key >= min_key);
		assert(key <= max_key);
	}
}

void check_range_partition(uint32_t *keys, uint64_t size, uint64_t *count,
			   uint64_t range_partitions, uint32_t *delim)
{	uint64_t r, p = 0;
	for (r = 0 ; r != range_partitions ; ++r) {
		check_range(&keys[p], count[r], r, delim);
		p += count[r];
	}
	assert(p == size);
}

uint64_t range_partition_to_blocks(uint32_t *keys, uint32_t *rids, uint64_t size,
	    uint32_t *keys_out, uint32_t *rids_out, uint32_t delim[],
	    volatile uint8_t *partition, uint64_t block_offset, uint64_t block_cap,
	    uint64_t *open_block_index, uint64_t *open_block_size)
{
	int i, partitions = 128;
	assert((15 & (uint64_t) keys) == 0);
	assert((15 & (uint64_t) rids) == 0);
	assert((63 & (uint64_t) keys_out) == 0);
	assert((63 & (uint64_t) rids_out) == 0);
	assert((size & 3) == 0);
	uint32_t *keys_end = &keys[size];
	// initialize buffers
	uint8_t block_cap_bits = log_2(block_cap);
	uint64_t block_cap_mask = block_cap - 1;
	uint64_t p = 0, q = 0;
	uint64_t blocks = partitions;
	uint64_t *buf = mamalloc((partitions << 4) * sizeof(uint64_t));
	for (i = 0 ; i != partitions ; ++i) {
		buf[(i << 4) | 15] = i << block_cap_bits;
		partition[i + block_offset] = i;
	}
	// main partition loop
	uint32_t *cdelim = mamalloc(128 * sizeof(uint32_t));
	uint64_t convert = 1 << 31;
	for (p = 0 ; p != 128 ; ++p)
		cdelim[p] = delim[p] - convert;
	__m128i c = _mm_set1_epi32(convert);
	__m128i d1 = _mm_set1_epi32(cdelim[15]);
	__m128i d2 = _mm_set1_epi32(cdelim[31]);
	__m128i d3 = _mm_set1_epi32(cdelim[47]);
	__m128i d4 = _mm_set1_epi32(cdelim[63]);
	__m128i d5 = _mm_set1_epi32(cdelim[79]);
	__m128i d6 = _mm_set1_epi32(cdelim[95]);
	__m128i d7 = _mm_set1_epi32(cdelim[111]);
	__m128i I = _mm_set1_epi32(~0); p = 0;
	while (keys != keys_end) {
		__m128i k = _mm_load_si128((__m128i*) keys);
		__m128i v = _mm_load_si128((__m128i*) rids);
		keys += 4; rids += 4;
		k = _mm_sub_epi32(k, c);
		__m128i e1 = _mm_cmpgt_epi32(k, d4);
		__m128i d26 = _mm_blendv_epi8(d2, d6, e1);
		__m128i d15 = _mm_blendv_epi8(d1, d5, e1);
		__m128i d37 = _mm_blendv_epi8(d3, d7, e1);
		__m128i e2 = _mm_cmpgt_epi32(k, d26);
		__m128i d1357 = _mm_blendv_epi8(d15, d37, e2);
		__m128i e3 = _mm_cmpgt_epi32(k, d1357);
		__m128i r = _mm_setzero_si128();
		r = _mm_sub_epi32(r, e1);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e2);
		r = _mm_add_epi32(r, r);
		r = _mm_sub_epi32(r, e3);
		r = _mm_slli_epi32(r, 4);
		for (i = 0 ; i != 4 ; ++i) {
			asm("movd	%1, %%eax" : "=a"(p) : "x"(r), "0"(p));
			__m128i di1 = _mm_load_si128((__m128i*) &cdelim[p]);
			__m128i di2 = _mm_load_si128((__m128i*) &cdelim[p + 4]);
			__m128i di3 = _mm_load_si128((__m128i*) &cdelim[p + 8]);
			__m128i di4 = _mm_load_si128((__m128i*) &cdelim[p + 12]);
			__m128i k_x4 = _mm_shuffle_epi32(k, 0);
			__m128i f1 = _mm_cmpgt_epi32(k_x4, di1);
			__m128i f2 = _mm_cmpgt_epi32(k_x4, di2);
			__m128i f3 = _mm_cmpgt_epi32(k_x4, di3);
			__m128i f4 = _mm_cmpgt_epi32(k_x4, di4);
			__m128i f12 = _mm_packs_epi32(f1, f2);
			__m128i f34 = _mm_packs_epi32(f3, f4);
			__m128i f = _mm_packs_epi16(f12, f34);
			f = _mm_xor_si128(f, I);
			asm("pmovmskb	%1, %%ebx" : "=b"(q) : "x"(f), "0"(q));
			asm("bsfl	%%ebx, %%ebx" : "=b"(q) : "0"(q) : "cc");
			p |= q;
#ifdef BG
			uint32_t key = keys[i - 4];
			assert(p == 0 || key > delim[p - 1]);
			assert(key <= delim[p]);
#endif
			// offset in the cache line pair
			uint64_t *src = &buf[p << 4];
			uint64_t index = src[15]++;
			uint64_t offset = index & 15;
			// pack and store
			__m128i kc = _mm_add_epi32(k_x4, c);
			__m128i kvxx = _mm_unpacklo_epi32(kc, v);
			_mm_storel_epi64((__m128i*) (float*) &src[offset], kvxx);
			if (offset == 15) {
				float *dest_x = (float*) &keys_out[index - 15];
				float *dest_y = (float*) &rids_out[index - 15];
				// load cache line from cache to 8 128-bit registers
				__m128 r0 = _mm_load_ps((float*) &src[0]);
				__m128 r1 = _mm_load_ps((float*) &src[2]);
				__m128 r2 = _mm_load_ps((float*) &src[4]);
				__m128 r3 = _mm_load_ps((float*) &src[6]);
				__m128 r4 = _mm_load_ps((float*) &src[8]);
				__m128 r5 = _mm_load_ps((float*) &src[10]);
				__m128 r6 = _mm_load_ps((float*) &src[12]);
				__m128 r7 = _mm_load_ps((float*) &src[14]);
				// split first column
				__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
				__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
				// stream first column
				_mm_stream_ps(&dest_x[0], x0);
				_mm_stream_ps(&dest_x[4], x1);
				_mm_stream_ps(&dest_x[8], x2);
				_mm_stream_ps(&dest_x[12],x3);
				// split second column
				__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
				__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
				// stream second column
				_mm_stream_ps(&dest_y[0], y0);
				_mm_stream_ps(&dest_y[4], y1);
				_mm_stream_ps(&dest_y[8], y2);
				_mm_stream_ps(&dest_y[12],y3);
				// restore overwritten pointer
				src[15] = ++index;
				// check if end of block
				if ((index & block_cap_mask) == 0) {
					// get next block
					src[15] = blocks << block_cap_bits;
					partition[block_offset + blocks] = p;
					blocks++;
#ifdef BG
					check_range(&keys_out[index - block_cap], block_cap, p, delim);
#endif
				}
			}
			// rotate
			r = _mm_shuffle_epi32(r, _MM_SHUFFLE(0, 3, 2, 1));
			k = _mm_shuffle_epi32(k, _MM_SHUFFLE(0, 3, 2, 1));
			v = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 3, 2, 1));
		}
	}
	free(cdelim);
	// flush remaining items from buffers to output
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t *src = &buf[i << 4];
		uint64_t index = src[15];
		uint64_t rem = index & 15;
		uint64_t off = 0;
		index -= rem;
		while (off != rem) {
			uint64_t pair = src[off++];
			keys_out[index] = (uint32_t) pair;
			rids_out[index++] = (uint32_t) (pair >> 32);
		}
		// set size of current block
		uint64_t block = index >> block_cap_bits;
		off = block << block_cap_bits;
		open_block_index[i] = block + block_offset;
		open_block_size[i] = index - off;
#ifdef BG
		check_range(&keys_out[off], index - off, i, delim);
#endif
	}
	return blocks;
}

void histogram(uint32_t *keys, uint64_t size, uint64_t *count,
	       uint8_t shift_bits, uint8_t radix_bits)
{
	uint32_t *keys_end = &keys[size];
	uint64_t partitions = 1 << radix_bits;
	uint64_t p, mask = partitions - 1;
	for (p = 0 ; p != partitions ; ++p)
		count[p] = 0;
	while ((15 & ((uint64_t) keys)) && keys != keys_end)
		count[(*keys++ >> shift_bits) & mask]++;
	__m128i m = _mm_cvtsi32_si128(mask);
	__m128i s = _mm_cvtsi32_si128(shift_bits);
	m = _mm_shuffle_epi32(m, 0);
	uint32_t *keys_aligned_end = &keys[(keys_end - keys) & ~3];
	uint64_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
	while (keys != keys_aligned_end) {
		__m128i k = _mm_load_si128((__m128i*) keys);
		keys += 4;
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
		__m128i h_s1 = _mm_shuffle_epi32(h, 1);
		__m128i h_s2 = _mm_shuffle_epi32(h, 2);
		__m128i h_s3 = _mm_shuffle_epi32(h, 3);
		asm("movd	%1, %%eax" : "=a"(p1) : "x"(h),    "0"(p1));
		asm("movd	%1, %%ebx" : "=b"(p2) : "x"(h_s1), "0"(p2));
		asm("movd	%1, %%ecx" : "=c"(p3) : "x"(h_s2), "0"(p3));
		asm("movd	%1, %%edx" : "=d"(p4) : "x"(h_s3), "0"(p4));
		count[p1]++;
		count[p2]++;
		count[p3]++;
		count[p4]++;
	}
	while (keys != keys_end)
		count[(*keys++ >> shift_bits) & mask]++;
}

void partition_ip(uint32_t *keys, uint32_t *rids, uint64_t size,
		  uint64_t *sizes, uint64_t *offsets,
		  uint8_t shift_bits, uint8_t radix_bits)
{
	uint64_t partitions = 1 << radix_bits;
	uint64_t mask = partitions - 1;
	uint64_t i, p, e;
	for (i = p = 0 ; p != partitions ; ++p) {
		i += sizes[p];
		offsets[p] = i;
	}
	assert(i == size);
	for (e = p = 0 ; sizes[p] == 0 ; ++p);
	do {
		uint32_t key = keys[e];
		uint32_t rid = rids[e];
		do {
			i = (key >> shift_bits) & mask;
			i = --offsets[i];
			uint32_t tmp_key = keys[i];
			uint32_t tmp_rid = rids[i];
			keys[i] = key;
			rids[i] = rid;
			key = tmp_key;
			rid = tmp_rid;
		} while (i != e);
		do {
			e += sizes[p++];
		} while (p != partitions && e == offsets[p]);
	} while (p != partitions);
}

typedef struct {
	uint64_t item_0;
	uint64_t end_index;
} extra_t;

uint64_t pair(uint32_t *x, uint32_t *y, uint64_t i)
{
	uint64_t r = y[i];
	return (r << 32) | x[i];
}

void partition_ip_buf(uint32_t *keys, uint32_t *rids, uint64_t size,
		      uint64_t *sizes, uint8_t shift_bits, uint8_t radix_bits)
{
	int i, j, partitions = 1 << radix_bits;
	uint64_t mask = (1 << radix_bits) - 1;
	assert((63 & (uint64_t) keys) == (63 & (uint64_t) rids));
	// allocate buffers
	extra_t *extra = malloc(partitions * sizeof(extra_t));
	uint64_t *buf = mamalloc((partitions << 4) * sizeof(uint64_t));
	// remove offset from alignment
	uint64_t virtual_add = (63 & (uint64_t) keys) >> 2;
	keys -= virtual_add;
	rids -= virtual_add;
	// initialize buffers
	uint64_t to = virtual_add;
	for (i = 0 ; i != partitions ; ++i) {
		uint64_t from = to;
		to += sizes[i];
		// if last block
		if (sizes[i] + (from & 15) <= 16)
			// load only specific items
			for (j = 0 ; j != sizes[i] ; ++j)
				buf[(i << 4) | j] = pair(keys, rids, j + from);
		// ending in half-full block
		else
			// load items in half-full last block plus extra items
			for (j = 0 ; j != 16 ; ++j)
				buf[(i << 4) | j] = pair(keys, rids, j + ((to - 1) & ~15));
		// store item zero and ending pointer
		extra[i].item_0 = buf[i << 4];
		extra[i].end_index = from;
		// if last block
		if (sizes[i] + (from & 15) <= 16)
			// set last cache line indicator
			buf[i << 4] = sizes[i] + ~ (uint64_t) 31;
		else
			// set running partition index
			buf[i << 4] = to;
	}
	// initial active cycle
	for (i = 0 ; sizes[i] == 0 ; ++i);
	uint64_t from = virtual_add;
	// initial active row
	uint64_t row = pair(keys, rids, from);
	uint64_t hash, index, mod_index, *addr;
	for (;;) {
		do {
			// compute hash
			hash = row;
			hash >>= shift_bits;
			hash &= mask;
			// get index and update pointer
			addr = &buf[hash << 4];
			index = --addr[0];
			mod_index = index & 15;
			// load previous key (if not zero offset)
			uint64_t temp = addr[mod_index];
			// store new index and item
			addr[mod_index] = row;
			// set the new active row
			row = temp;
			// if not end of cache line restart loop (7/8 times taken)
		} while (mod_index);
		// check if partition is now full (almost never taken)
		if (index == ~ (uint64_t) 31) {
			// restore zero item (useful when cycle not closed)
			row = extra[hash].item_0;
			// store zero item and restore end indicator
			extra[hash].item_0 = addr[0];
			addr[0] = ~ (uint64_t) 31;
			// if cycle closed
			if (i == hash) {
				// find next non filled partition
				while (buf[i << 4] == ~ (uint64_t) 31) {
					from += sizes[i++];
					if (i == partitions) break;
				}
				if (i == partitions) break;
				// start new cycle
				row = pair(keys, rids, from);
			}
			continue;
		}
		float *keys_out = (float*) &keys[index - 16];
		float *rids_out = (float*) &rids[index - 16];
		// load first half buffered cache line
		__m128 r0 = _mm_load_ps((float*) &addr[0]);
		__m128 r1 = _mm_load_ps((float*) &addr[2]);
		__m128 r2 = _mm_load_ps((float*) &addr[4]);
		__m128 r3 = _mm_load_ps((float*) &addr[6]);
		// split half line
		__m128 x0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 y0 = _mm_shuffle_ps(r0, r1, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 x1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 y1 = _mm_shuffle_ps(r2, r3, _MM_SHUFFLE(3, 1, 3, 1));
		// stream half line
		_mm_stream_ps(&keys_out[16], x0);
		_mm_stream_ps(&rids_out[16], y0);
		_mm_stream_ps(&keys_out[20], x1);
		_mm_stream_ps(&rids_out[20], y1);
		// load second half buffered cache line
		__m128 r4 = _mm_load_ps((float*) &addr[8]);
		__m128 r5 = _mm_load_ps((float*) &addr[10]);
		__m128 r6 = _mm_load_ps((float*) &addr[12]);
		__m128 r7 = _mm_load_ps((float*) &addr[14]);
		// split half line
		__m128 x2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 y2 = _mm_shuffle_ps(r4, r5, _MM_SHUFFLE(3, 1, 3, 1));
		__m128 x3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(2, 0, 2, 0));
		__m128 y3 = _mm_shuffle_ps(r6, r7, _MM_SHUFFLE(3, 1, 3, 1));
		// stream half line
		_mm_stream_ps(&keys_out[24], x2);
		_mm_stream_ps(&rids_out[24], y2);
		_mm_stream_ps(&keys_out[28], x3);
		_mm_stream_ps(&rids_out[28], y3);
		// load half items from columns
		__m128i x4 = _mm_load_si128((__m128i*) &keys_out[0]);
		__m128i y4 = _mm_load_si128((__m128i*) &rids_out[0]);
		__m128i x5 = _mm_load_si128((__m128i*) &keys_out[4]);
		__m128i y5 = _mm_load_si128((__m128i*) &rids_out[4]);
		// transpose first half
		__m128i p0 = _mm_unpacklo_epi32(x4, y4);
		__m128i p1 = _mm_unpackhi_epi32(x4, y4);
		__m128i p2 = _mm_unpacklo_epi32(x5, y5);
		__m128i p3 = _mm_unpackhi_epi32(x5, y5);
		// store half in array
		_mm_store_si128((__m128i*) &addr[0], p0);
		_mm_store_si128((__m128i*) &addr[2], p1);
		_mm_store_si128((__m128i*) &addr[4], p2);
		_mm_store_si128((__m128i*) &addr[6], p3);
		// load half items from columns
		__m128i x6 = _mm_load_si128((__m128i*) &keys_out[8]);
		__m128i y6 = _mm_load_si128((__m128i*) &rids_out[8]);
		__m128i x7 = _mm_load_si128((__m128i*) &keys_out[12]);
		__m128i y7 = _mm_load_si128((__m128i*) &rids_out[12]);
		// transpose second half
		__m128i p4 = _mm_unpacklo_epi32(x6, y6);
		__m128i p5 = _mm_unpackhi_epi32(x6, y6);
		__m128i p6 = _mm_unpacklo_epi32(x7, y7);
		__m128i p7 = _mm_unpackhi_epi32(x7, y7);
		// store half in array
		_mm_store_si128((__m128i*) &addr[8],  p4);
		_mm_store_si128((__m128i*) &addr[10], p5);
		_mm_store_si128((__m128i*) &addr[12], p6);
		_mm_store_si128((__m128i*) &addr[14], p7);
		// set the new active row by restoring zero item (not the pointer)
		row = extra[hash].item_0;
		// read the end of the partition
		uint64_t end = extra[hash].end_index;
		// save the new zero item
		extra[hash].item_0 = addr[0];
		// set the index in offset zero
		addr[0] = index;
		// check if not last block (almost always taken)
		uint64_t rem = index - end;
		if (rem > 16) continue;
		// move items towards the start of the cache line
		addr[0] = extra[hash].item_0;
		for (j = 0 ; j != rem ; ++j)
			addr[j] = addr[16 - rem + j];
		extra[hash].item_0 = addr[0];
		// set last block index indicator
		addr[0] = rem + ~ (uint64_t) 31;
	}
	// write out the last buffer data in order
	from = virtual_add;
	for (i = 0 ; i != partitions ; ++i) {
		if (sizes[i] == 0) continue;
		uint64_t rem = 16 - (from & 15);
		if (rem > sizes[i]) rem = sizes[i];
		// restore first item
		buf[i << 4] = extra[i].item_0;
		// store specific items to output
		for (j = 0 ; j != rem ; ++j) {
			keys[from + j] = (uint32_t)  buf[(i << 4) | j];
			rids[from + j] = (uint32_t) (buf[(i << 4) | j] >> 32);
		}
		from += sizes[i];
	}
	free(buf);
	free(extra);
}

inline void combsort(uint32_t *keys, uint32_t *rids, uint64_t size)
{
	static const float shrink = 0.77;
	uint64_t gap = size * shrink;
	for (;;) {
		uint64_t i = 0;
		uint64_t j = gap;
		uint64_t done = 1;
		do {
			uint32_t ki = keys[i];
			uint32_t kj = keys[j];
			if (ki > kj) {
				uint32_t r = rids[i];
				rids[i] = rids[j];
				rids[j] = r;
				keys[i] = kj;
				keys[j] = ki;
				done = 0;
			}
			i++;
			j++;
		} while (j != size);
		if (gap > 1) gap *= shrink;
		else if (done) break;
	}
}

inline void insertsort(uint32_t *keys, uint32_t *rids, uint64_t size)
{
	if (size <= 1) return;
	uint32_t prev_key = keys[0];
	uint64_t i = 1;
	do {
		uint32_t next_key = keys[i];
		if (next_key >= prev_key)
			prev_key = next_key;
		else {
			uint32_t next_rid = rids[i];
			uint32_t temp_key = prev_key;
			uint64_t j = i - 1;
			do {
				rids[j + 1] = rids[j];
				keys[j + 1] = temp_key;
				if (j-- == 0) break;
				temp_key = keys[j];
			} while (next_key < temp_key);
			keys[j + 1] = next_key;
			rids[j + 1] = next_rid;
		}
	} while (++i != size);
}

void local_radixsort(uint32_t *keys, uint32_t *rids, uint64_t size,
		     int8_t *bits, int8_t *buffered, int depth,
		     uint64_t **hist, uint64_t **offsets)
{
	if (size <= 20) {
		insertsort(keys, rids, size);
		return;
	}
	int8_t in_buf = buffered[depth];
	if (in_buf < 0) {
		combsort(keys, rids, size);
		return;
	}
	int8_t shift_bits = bits[depth + 1];
	int8_t radix_bits = bits[depth] - shift_bits;
	uint64_t partitions = 1 << radix_bits;
	histogram(keys, size, hist[depth], shift_bits, radix_bits);
	if (in_buf)
		partition_ip_buf(keys, rids, size, hist[depth], shift_bits, radix_bits);
	else
		partition_ip(keys, rids, size, hist[depth], offsets[depth], shift_bits, radix_bits);
	if (shift_bits == 0) return;
	uint64_t i, j, *h = hist[depth];
	for (i = j = 0 ; i != partitions ; ++i) {
		local_radixsort(&keys[j], &rids[j], h[i],
				bits, buffered, depth + 1, hist, offsets);
		j += h[i];
	}
}

void partition_keys(uint32_t *keys, uint32_t *keys_out, uint64_t size,
		    uint64_t **hist, uint8_t shift_bits, uint8_t radix_bits,
		    int thread_id, int threads, pthread_barrier_t *barrier)
{
	// inputs and outputs must be aligned
	assert(0 == (15 & (size_t) keys));
	assert(0 == (63 & (size_t) keys_out));
	assert((size & 3) == 0);
	// set sub-array for current thread
	uint64_t local_size = (size / threads) & ~15;
	uint32_t *local_keys = &keys[local_size * thread_id];
	if (thread_id + 1 == threads)
		local_size = size - local_size * thread_id;
	// initialize histogram
	uint64_t i, j, p; int t;
	uint64_t partitions = 1 << radix_bits;
	uint64_t *local_hist = hist[thread_id];
	for (p = 0 ; p != partitions ; ++p)
		local_hist[p] = 0;
	// main histogram loop
	__m128i s = _mm_set_epi32(0, 0, 0, shift_bits);
	__m128i m = _mm_set1_epi32((1 << radix_bits) - 1);
	for (i = p = 0 ; i != local_size ; i += 4) {
		__m128i k = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
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
		__m128i k = _mm_load_si128((__m128i*) &local_keys[i]);
		__m128i h = _mm_srl_epi32(k, s);
		h = _mm_and_si128(h, m);
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
	free(index);
	free(buf);
}

inline void copy(int32_t *dst, const int32_t *src, uint64_t size)
{
	if (!size) return;
	const int32_t *src_end = &src[size];
	do {
		_mm_stream_si32(dst++, *src++);
	} while (src != src_end);
}

inline void move(int32_t *dst, int32_t *src, uint64_t size)
{
	int64_t diff = dst - src;
	uint64_t abs_diff = diff < 0 ? -diff : diff;
	// no overlap => copy everything
	if (abs_diff >= size)
		copy(dst, src, size);
	// move left => copy from right
	else if (dst < src)
		copy(dst, &dst[size], abs_diff);
	// move right => copy from left
	else if (dst > src)
		copy(&src[size], src, abs_diff);
}

void block_store(int32_t *dst, const int32_t *src, uint64_t size)
{
	assert(size > 0 && (size & 15) == 0);
	const int32_t *src_end = &src[size];
	do {
		__m128i x = _mm_load_si128((__m128i*) src);
		_mm_store_si128((__m128i*) dst, x);
		src += 4;
		dst += 4;
	} while (src != src_end);
}

void block_stream(int32_t *dst, const int32_t *src, uint64_t size)
{
	assert(size > 0 && (size & 15) == 0);
	const int32_t *src_end = &src[size];
	do {
		__m128i x = _mm_load_si128((__m128i*) src);
		_mm_stream_si128((__m128i*) dst, x);
		src += 4;
		dst += 4;
	} while (src != src_end);
}

void block_swap(int32_t *mem, int32_t *buf, uint64_t size)
{
	assert(size > 0 && (size & 15) == 0);
	const int32_t *mem_end = &mem[size];
	do {
		__m128i x = _mm_load_si128((__m128i*) mem);
		__m128i y = _mm_load_si128((__m128i*) buf);
		_mm_store_si128((__m128i*) buf, x);
		_mm_stream_si128((__m128i*) mem, y);
		mem += 4;
		buf += 4;
	} while (mem != mem_end);
}

void swap(int32_t *x, int32_t *y, uint64_t size)
{
	if (!size) return;
	int32_t *x_end = &x[size];
	do {
		int32_t x_val = *x;
		int32_t y_val = *y;
		_mm_stream_si32(x++, y_val);
		_mm_stream_si32(y++, x_val);
	} while (x != x_end);
}

uint64_t combine(uint32_t **keys, uint32_t **rids, uint64_t *size,
		 uint32_t **m_keys, uint32_t **m_rids, uint64_t *m_size,
		 uint64_t total, int m_total, uint64_t cap)
{
	// copy new items in the blocks
	uint64_t m = 0;
	while (m != m_total) {
		if (m_size[m] == 0) {
			m++;
			continue;
		}
		uint64_t i, max_i = ~0;
		// find largest non-full block
		for (i = 0 ; i != total ; ++i)
			if (size[i] != cap && (max_i == ~0 || size[i] > size[max_i]))
				max_i = i;
		// copy data from new to largest
		uint64_t n = min(cap - size[max_i], m_size[m]);
		m_size[m] -= n;
		copy(&keys[max_i][size[max_i]], &m_keys[m][m_size[m]], n);
		copy(&rids[max_i][size[max_i]], &m_rids[m][m_size[m]], n);
		size[max_i] += n;
	}
#ifdef BG
	for (m = 0 ; m != m_total ; ++m)
		assert(m_size[m] == 0);
#endif
	// compact the blocks
	for (;;) {
		uint64_t i, max_i = ~0, min_i = ~0;
		// find largest non-empty non-full block (smallest index in tie)
		for (i = 0 ; i != total ; ++i)
			if (size[i] != cap && size[i] && (max_i == ~0 || size[i] > size[max_i]))
				max_i = i;
		// find smallest non-empty non-full block (largest index in tie)
		for (i = 0 ; i != total ; ++i)
			if (size[i] != cap && size[i] && (min_i == ~0 || size[i] <= size[min_i]))
				min_i = i;
		// if same return it
		if (max_i == min_i) {
#ifdef BG
			for (i = 0 ; i != total ; ++i)
				assert(size[i] == 0 || size[i] == cap || i == max_i);
			assert(max_i == ~0 || (size[max_i] > 0 && size[max_i] < cap));
#endif
			return max_i;
		}
		assert(size[max_i] != 0);
		assert(size[min_i] != 0);
		// copy data from smallest to largest
		uint64_t n = min(cap - size[max_i], size[min_i]);
		size[min_i] -= n;
		copy(&keys[max_i][size[max_i]], &keys[min_i][size[min_i]], n);
		copy(&rids[max_i][size[max_i]], &rids[min_i][size[min_i]], n);
		size[max_i] += n;
	}
}

uint64_t inject(uint32_t *keys, uint32_t *rids, uint64_t *size,
		uint64_t parts, uint64_t block_cap,
		uint32_t **e_keys, uint32_t **e_rids, uint64_t *e_size)
{
	uint64_t p, before_total = 0, after_total = 0;
	for (p = 0 ; p != parts ; ++p) {
		before_total += size[p] * block_cap;
		after_total += e_size[p];
	}
	after_total += before_total;
	uint64_t total = after_total;
	for (p = parts - 1 ; p != ~0 ; --p) {
		// compute locations before and after
		before_total -= size[p] * block_cap;
		after_total -= size[p] * block_cap + e_size[p];
		// move items to make space
		move(&keys[after_total], &keys[before_total], size[p] * block_cap);
		move(&rids[after_total], &rids[before_total], size[p] * block_cap);
		// copy items in space created
		copy(&keys[after_total + size[p] * block_cap], e_keys[p], e_size[p]);
		copy(&rids[after_total + size[p] * block_cap], e_rids[p], e_size[p]);
	}
	assert(after_total == 0 && before_total == 0);
	return total;
}

void extract_delimiters(uint32_t *sample, uint64_t sample_size, uint32_t *delimiter)
{
	uint64_t i, parts = 0;
	while (delimiter[parts] != ~0) parts++;
	double percentile = sample_size * 1.0 / (parts + 1);
	for (i = 0 ; i != parts ; ++i) {
		uint64_t index = percentile * (i + 1) - 0.001;
		delimiter[i] = sample[index];
		// search repetitions in sample
		uint64_t start, end;
		for (start = index ; start ; --start)
			if (sample[start] != delimiter[i]) break;
		for (end = index ; end != sample_size ; ++end)
			if (sample[end] != delimiter[i]) break;
		// if more repetitions after, don't include
		if (index - start < end - index && delimiter[i])
			delimiter[i]--;
	}
}

inline uint8_t ceil_log(uint64_t x)
{
	uint8_t p = 0;
	uint64_t o = 1;
	while ((o << p) < x) p++;
	return p;
}

inline uint64_t ceil_div(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

int schedule_passes(uint64_t size, int8_t bits, int8_t *radix_bits, int8_t *buffered)
{
	int i, p = 0;
	uint64_t cache_limit = 12500;
	uint64_t pieces = ceil_div(size, cache_limit);
	int8_t log_pieces = ceil_log(ceil_div(size, cache_limit));
	int8_t initial_log_pieces = log_pieces;
	assert(log_pieces < bits);
	// split up to 32-way using in-cache variant
	if (size <= cache_limit) ;
	else if (log_pieces <= 5) {
		if (log_pieces < 3) log_pieces = 3;
		if (log_pieces > bits) log_pieces = bits;
		buffered[p] = 0;
		radix_bits[p++] = log_pieces;
	// split up to 512-way using out-of-cache variant
	} else if (log_pieces <= 9) {
		buffered[p] = 1;
		radix_bits[p++] = log_pieces;
	// split up-to 2^18-way using out-of-cache : out-of-cache variant
	} else if (log_pieces <= 18) {
		buffered[p] = 1;
		radix_bits[p++] = log_pieces >> 1;
		buffered[p] = 1;
		radix_bits[p++] = (log_pieces + 1) >> 1;
	// split up-to 2^27-way using out-of-cache : out-of-cache : out-of-cache variant
	} else if (log_pieces <= 27) {
		buffered[p] = 1;
		radix_bits[p++] = log_pieces / 3;
		log_pieces -= log_pieces / 3;
		buffered[p] = 1;
		radix_bits[p++] = log_pieces >> 1;
		buffered[p] = 1;
		radix_bits[p++] = (log_pieces + 1) >> 1;
	} else assert(log_pieces <= 27);
	// remove all passes
	for (i = 0 ; i != p ; ++i) {
		size >>= radix_bits[i];
		bits -= radix_bits[i];
	}
	assert(size <= cache_limit);
	i = ceil_log(size) - 2;
	if (i > bits) i = bits;
	bits -= i;
	buffered[p] = 0;
	radix_bits[p++] = i;
	if (bits <= p) {
		for (i = p - bits ; i != p ; ++i)
			radix_bits[i]++;
		bits = 0;
	}
	// seal the sequence
	buffered[p] = -1;
	radix_bits[p] = bits;
	return p;
}

typedef struct {
	uint32_t **keys;
	uint32_t **rids;
	uint64_t *size;
	// sync data
	volatile uint64_t *moved;
	volatile uint64_t *offline_up;
	volatile uint64_t *offline_down;
	volatile uint64_t *ranges_compacted;
	volatile uint64_t *ranges_closed;
	// first phase last
	uint64_t ***thread_open_block_index;
	uint64_t ***thread_open_block_size;
	uint64_t **thread_total_blocks;
	volatile uint64_t **thread_unused_blocks;
	// block map of items
	volatile int8_t **block_map;
	uint64_t **count_blocks;
	uint64_t *numa_blocks;
	volatile uint64_t *more_blocks;
	uint64_t block_cap;
	// copied items
	uint32_t ***first_keys;
	uint32_t ***first_rids;
	uint64_t **first_size;
	// half block keys
	uint32_t **half_block_keys;
	uint32_t **half_block_rids;
	uint64_t *half_block_size;
	// last items
	uint64_t *open;
	uint64_t **open_block;
	// buffer space
	uint32_t **keys_space;
	uint32_t **rids_space;
	// misc info
	int *numa_dest;
	uint64_t **max_block_index;
	uint64_t *numa_range_of_ranges;
	double fudge;
	// sample
	uint32_t *sample;
	uint32_t *sample_buf;
	uint64_t sample_size;
	uint64_t **sample_hist;
	// more info
	int *numa_node;
	int *cpu;
	int numa;
	int threads;
	int max_numa;
	int max_threads;
	pthread_barrier_t *global_barrier;
	pthread_barrier_t **local_barrier;
	pthread_barrier_t *sample_barrier;
} global_data_t;

typedef struct {
	int id;
	uint64_t seed;
	uint64_t checksum;
	uint64_t sample_time;
	uint64_t partition_first_time;
	uint64_t partition_blocks_time;
	uint64_t combine_time;
	uint64_t compact_time;
	uint64_t balance_time;
	uint64_t block_swap_online_time;
	uint64_t block_swap_offline_time;
	uint64_t inject_time;
	uint64_t local_sort_time;
	uint64_t local_sort_tuples;
	global_data_t *global;
} thread_data_t;

void *sort_thread(void *arg)
{
	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	int i, j, t, n, id = a->id;
	int lb = 0, gb = 0;
	uint64_t p, b, q, l;
	int numa = d->numa;
	int numa_node = d->numa_node[id];
	int numa_dst, numa_src;
	int threads = d->threads;
	int threads_per_numa = threads / numa;
	pthread_barrier_t *local_barrier = d->local_barrier[numa_node];
	pthread_barrier_t *global_barrier = d->global_barrier;
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
	// block info
	int range_partitions = 128;
	uint64_t block_cap = d->block_cap;
	uint8_t block_cap_bits = log_2(block_cap);
	// inputs, outputs and size
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = (numa_size / threads_per_numa) & ~(block_cap - 1);
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - offset;
	// sample keys from local data
	uint32_t *keys = &d->keys[numa_node][offset];
	uint64_t tim = micro_time();
	assert((d->sample_size & 3) == 0);
	uint64_t sample_size = (d->sample_size / threads) & ~15;
	uint32_t *sample = &d->sample[sample_size * id];
	if (id + 1 == threads)
		sample_size = d->sample_size - sample_size * id;
	rand64_t *gen = rand64_init(a->seed);
	for (p = 0 ; p != sample_size ; ++p)
		sample[p] = keys[mulhi(rand64_next(gen), size)];
#ifdef BG
	if (id == 0) fprintf(stderr, "Sampling done!\n");
#endif
	// (in-parallel) LSB radix-sort the sample
	partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 0, 8,
	               id, threads, &global_barrier[gb]);
	partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 8, 8,
	               id, threads, &global_barrier[gb + 3]);
	partition_keys(d->sample, d->sample_buf, d->sample_size, d->sample_hist, 16, 8,
	               id, threads, &global_barrier[gb + 6]);
	partition_keys(d->sample_buf, d->sample, d->sample_size, d->sample_hist, 24, 8,
	               id, threads, &global_barrier[gb + 9]);
	gb += 12;
	tim = micro_time() - tim;
	a->sample_time = tim;
	// extract delimiters from sample
	uint64_t half_range_partitions = range_partitions >> 1;
	uint32_t *numa_delimiter = malloc(numa * sizeof(uint32_t));
	uint32_t *range_delimiter = mamalloc(range_partitions * sizeof(uint32_t));
	uint32_t *thread_delimiter = calloc(half_range_partitions, sizeof(uint32_t));
	thread_delimiter[half_range_partitions - 1] = ~0;
	extract_delimiters(d->sample, d->sample_size, thread_delimiter);
	for (p = 0 ; p != half_range_partitions ; ++p)
		range_delimiter[p] = thread_delimiter[p];
	// numa numa_delimiters
	q = half_range_partitions / numa;
	for (p = 0 ; p != numa ; ++p)
		numa_delimiter[p] = range_delimiter[(p + 1) * q - 1];
	// radix ranges
	uint8_t log_half_range_partitions = log_2(range_partitions) - 1;
	for (p = 0 ; p != half_range_partitions ; ++p) {
		uint64_t delim = (p << (32 - log_half_range_partitions)) - 1;
		range_delimiter[p + half_range_partitions] = delim;
	}
	qsort(range_delimiter, range_partitions, sizeof(uint32_t), uint32_compare);
	// compute total and local blocks
	uint64_t numa_blocks = (numa_size + block_cap - 1) / block_cap;
	uint64_t local_blocks = size / block_cap;
	uint64_t prev_blocks = offset / block_cap;
	assert(local_blocks >= range_partitions);
	// start locations
	keys = &d->keys[numa_node][prev_blocks << block_cap_bits];
	uint32_t *rids = &d->rids[numa_node][prev_blocks << block_cap_bits];
	// allocate space for blocks
	uint64_t alloced_size = d->size[numa_node] * d->fudge;
	uint64_t alloced_blocks = alloced_size >> block_cap_bits;
	uint64_t max_local_blocks = 1 + numa_blocks +
				    threads_per_numa * range_partitions;
	assert(max_local_blocks <= alloced_blocks);
	if (!numa_local_id) {
		int8_t *block_map = malloc(alloced_blocks * sizeof(uint8_t));
		d->numa_blocks[numa_node] = numa_blocks;
		d->block_map[numa_node] = block_map;
	}
	// synchronize local nodes (to get block space)
	pthread_barrier_wait(&local_barrier[lb++]);
	// set block map
	volatile int8_t **block_map = d->block_map;
	volatile int8_t *local_block_map = d->block_map[numa_node];
	// reset the indicators of free blocks
	uint64_t max_blocks_size = max_local_blocks / threads_per_numa;
	uint64_t max_blocks_offset = max_blocks_size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		max_blocks_size = max_local_blocks - max_blocks_offset;
	memset((void*) &local_block_map[max_blocks_offset], -1, max_blocks_size);
	// determine numa destination per range partition
	if (!id) {
		int *numa_dest = malloc(range_partitions * sizeof(int));
		for (p = 0, n = 0 ; n != numa - 1 ; ++n) {
			q = binary_search_32(range_delimiter, range_partitions,
					     numa_delimiter[n]);
			assert(range_delimiter[q] == numa_delimiter[n]);
			for (; p <= q ; ++p)
				numa_dest[p] = n;
		}
		for (; p != range_partitions ; ++p)
			numa_dest[p] = numa - 1;
		d->numa_dest = numa_dest;
	}
	// space for counts
	uint64_t *count = calloc(range_partitions, sizeof(uint64_t));
	// allocate space for first read
	uint32_t *keys_space = mamalloc(block_cap * range_partitions * sizeof(uint32_t));
	uint32_t *rids_space = mamalloc(block_cap * range_partitions * sizeof(uint32_t));
	int8_t *ranges = mamalloc(block_cap * range_partitions * sizeof(int8_t));
	// range partition histogram of the first items and save destinations
	tim = micro_time();
	uint64_t copy_part = min(size, block_cap * range_partitions);
	range_histogram(keys, ranges, copy_part, count, range_delimiter);
	// (range) partition from destinations and store in buffer space
	partition_known(keys, rids, ranges, copy_part, count,
			keys_space, rids_space, range_partitions);
	tim = micro_time() - tim;
	a->partition_first_time = tim;
	// broadcast extra items
	uint64_t o = 0;
	for (p = 0 ; p != range_partitions ; ++p) {
		d->first_keys[p][id] = &keys_space[o];
		d->first_rids[p][id] = &rids_space[o];
		d->first_size[p][id] = count[p];
		o += count[p];
	}
	assert(o == copy_part);
	// synchronize local nodes (to get block space)
	pthread_barrier_wait(&local_barrier[lb++]);
	free(count);
	// range partition the data and store back in block fashion
	uint32_t *mid_keys = &keys[copy_part];
	uint32_t *mid_rids = &rids[copy_part];
	uint64_t *open_block_index = malloc(range_partitions * sizeof(uint64_t));
	uint64_t *open_block_size = malloc(range_partitions * sizeof(uint64_t));
	tim = micro_time();
	b = range_partition_to_blocks(mid_keys, mid_rids, size - copy_part, keys, rids,
				      range_delimiter, local_block_map, prev_blocks,
				      block_cap, open_block_index, open_block_size);
	tim = micro_time() - tim;
	a->partition_blocks_time = tim;
	// used and unused blocks of local node
	d->thread_total_blocks[numa_node][numa_local_id] = local_blocks;
	d->thread_unused_blocks[numa_node][numa_local_id] = local_blocks - b;
	// broadcast info on last blocks
	d->thread_open_block_index[numa_node][numa_local_id] = open_block_index;
	d->thread_open_block_size[numa_node][numa_local_id] = open_block_size;
	// synchronize all nodes
	pthread_barrier_wait(d->sample_barrier);
#ifdef BG
	// check data in blocks
	if (!numa_local_id) {
		for (b = 0 ; b != max_local_blocks ; ++b) {
			int8_t r = local_block_map[b];
			if (r < 0) continue;
			uint64_t block_size = block_cap;
			for (t = 0 ; t != threads_per_numa ; ++t)
				if (d->thread_open_block_index[numa_node][t][r] == b) {
					assert(block_size == block_cap);
					block_size = d->thread_open_block_size[numa_node][t][r];
				}
			check_range(&keys[b << block_cap_bits], block_size, r, range_delimiter);
		}
		fprintf(stderr, "Block ranges checked %d / %d\n", numa_node + 1, numa);
	}
	pthread_barrier_wait(&global_barrier[gb++]);
#endif
	d->max_block_index[id] = calloc(numa, sizeof(uint64_t));
	tim = micro_time();
	for (;;) {
		// skip other partitition data
		p = __sync_fetch_and_add(d->ranges_compacted, 1);
		if (p >= range_partitions) break;
		// compute total size of last items in current range partition
		uint64_t range_size = 0;
		for (n = 0 ; n != numa ; ++n)
			for (t = 0 ; t != threads_per_numa ; ++t)
				range_size += d->thread_open_block_size[n][t][p];
		for (t = 0 ; t != threads ; ++t)
			range_size += d->first_size[p][t];
		// how many blocks does partition need in total
		uint64_t range_blocks = (range_size + block_cap - 1) / block_cap;
		uint64_t extra_blocks = max(threads, range_blocks);
		// allocate the extra blocks required
		int *extra_block_numa = malloc(extra_blocks * sizeof(int));
		uint64_t *extra_block_index = malloc(extra_blocks * sizeof(uint64_t));
		uint64_t *extra_block_size =  malloc(extra_blocks * sizeof(uint64_t));
		uint32_t **extra_block_keys = malloc(extra_blocks * sizeof(uint32_t*));
		uint32_t **extra_block_rids = malloc(extra_blocks * sizeof(uint32_t*));
		uint64_t e = 0;
		// include last blocks of each thread first
		for (n = 0 ; n != numa ; ++n)
			for (t = 0 ; t != threads_per_numa ; ++t) {
				b = d->thread_open_block_index[n][t][p];
				extra_block_keys[e] = &d->keys[n][b << block_cap_bits];
				extra_block_rids[e] = &d->rids[n][b << block_cap_bits];
				extra_block_index[e] = b;
				extra_block_size[e] = d->thread_open_block_size[n][t][p];
				extra_block_numa[e++] = n;
			}
		// loop through block map and add more empty blocks
		for (n = 0 ; n != numa && e != extra_blocks ; ++n)
			for (t = 0 ; t != threads_per_numa && e != extra_blocks ; ++t) {
				if (!d->thread_unused_blocks[n][t]) continue;
				// try to use empty block
				b = d->thread_total_blocks[n][t] - d->thread_unused_blocks[n][t];
				if (!__sync_bool_compare_and_swap(&block_map[n][b], -1, p))
					continue;
				// save block
				__sync_fetch_and_sub(&d->thread_unused_blocks[n][t], 1);
				extra_block_keys[e] = &d->keys[n][b << block_cap_bits];
				extra_block_rids[e] = &d->rids[n][b << block_cap_bits];
				extra_block_index[e] = b;
				extra_block_size[e] = 0;
				extra_block_numa[e++] = n;
			}
		// append more blocks until you have enough space
		while (e != extra_blocks) {
			// find numa with minimum blocks
			int min_more_blocks_numa = 0;
			uint64_t min_more_blocks = d->more_blocks[0];
			for (n = 1 ; n != numa ; ++n)
				if (d->more_blocks[n] < min_more_blocks) {
					min_more_blocks_numa = n;
					min_more_blocks = d->more_blocks[n];
				}
			n = min_more_blocks_numa;
			b = d->numa_blocks[n] +
			    __sync_fetch_and_add(&d->more_blocks[n], 1);
			// try to allocate extra block
			assert(block_map[n][b] == -1);
			block_map[n][b] = p;
			// save block
			assert(b < d->numa_blocks[n] + 1 +
				   range_partitions * threads_per_numa);
			extra_block_keys[e] = &d->keys[n][b << block_cap_bits];
			extra_block_rids[e] = &d->rids[n][b << block_cap_bits];
			extra_block_index[e] = b;
			extra_block_size[e] = 0;
			extra_block_numa[e++] = n;
		}
		// fill data from copied out items and move data around in the blocks
		e = combine(extra_block_keys, extra_block_rids, extra_block_size,
			    d->first_keys[p], d->first_rids[p], d->first_size[p],
			    extra_blocks, threads, block_cap);
		// copy out half-full block
		if (e != ~0) {
			uint64_t last_size = extra_block_size[e];
			// allocate space and copy half full block
			d->half_block_keys[p] = malloc(last_size * sizeof(uint32_t));
			d->half_block_rids[p] = malloc(last_size * sizeof(uint32_t));
			copy(d->half_block_keys[p], extra_block_keys[e], last_size);
			copy(d->half_block_rids[p], extra_block_rids[e], last_size);
			d->half_block_size[p] = last_size;
			extra_block_size[e] = 0;
		}
		// clean up the partition chain
		for (e = 0 ; e != extra_blocks ; ++e)
			// remove empty blocks
			if (extra_block_size[e] == 0) {
				n = extra_block_numa[e];
				b = extra_block_index[e];
				block_map[n][b] = -2;
			// update largest known non-empty block
			} else {
				assert(extra_block_size[e] == block_cap);
				n = extra_block_numa[e];
				b = extra_block_index[e];
				if (b > d->max_block_index[id][n])
					d->max_block_index[id][n] = b;
			}
		// free buffers
		free(extra_block_keys);
		free(extra_block_rids);
		free(extra_block_index);
		free(extra_block_size);
		free(extra_block_numa);
	}
	tim = micro_time() - tim;
	a->combine_time = tim;
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
	// compute extra items of current numa node
	uint64_t extra_numa_size = 0;
	for (p = 0 ; p != range_partitions ; ++p)
		if (d->numa_dest[p] == numa_node)
			extra_numa_size += d->half_block_size[p];
	// figure out local max block index from all threads
	uint64_t max_block_index = 0;
	for (t = 0 ; t != threads ; ++t)
		if (max_block_index < d->max_block_index[t][numa_node])
			max_block_index = d->max_block_index[t][numa_node];
	assert(max_block_index != 0);
	// sync globally
	pthread_barrier_wait(&global_barrier[gb++]);
#ifdef BG
	if (!numa_local_id) {
		fprintf(stderr, "NUMA %d max block (empty & full): %ld\n",
				 numa_node, max_block_index);
		for (b = 0 ; b != max_local_blocks ; ++b) {
			int8_t r = local_block_map[b];
			if (r < 0) continue;
			check_range(&keys[b << block_cap_bits], block_cap, r, range_delimiter);
		}
		fprintf(stderr, "Full blocks checked %d / %d\n", numa_node + 1, numa);
	}
	pthread_barrier_wait(&global_barrier[gb++]);
	if (!numa_local_id) {
		for (p = 0 ; p != range_partitions ; ++p)
			check_range(d->half_block_keys[p], d->half_block_size[p], p, range_delimiter);
		fprintf(stderr, "Half blocks checked %d / %d\n", numa_node + 1, numa);
	}
	pthread_barrier_wait(&global_barrier[gb++]);
#endif
	// search for first empty and swap with last full block
	numa_blocks = max_block_index + 1;
	assert(numa_blocks <= alloced_blocks);
	// pointers to blocks to be swapped
	uint64_t max_full = max_block_index;
	uint64_t min_empty = ~0;
	// keep swap pairs
	uint64_t swap_pair_cap = 32;
	uint64_t swap_pair_size = 0;
	uint64_t *swap_min_index = NULL;
	uint64_t *swap_max_index = NULL;
	if (!numa_local_id) {
		swap_min_index = malloc(swap_pair_cap * sizeof(uint64_t));
		swap_max_index = malloc(swap_pair_cap * sizeof(uint64_t));
	}
	keys = d->keys[numa_node];
	rids = d->rids[numa_node];
	tim = micro_time();
	for (;;) {
		// find next empty block (from start)
		for (b = min_empty + 1 ; b != max_full ; ++b)
			if (local_block_map[b] < 0) {
				min_empty = b;
				break;
			}
		if (b == max_full) break;
		// update total blocks
		assert(min_empty < max_full);
		// add move in list
		if (!numa_local_id) {
			if (swap_pair_size == swap_pair_cap) {
				swap_pair_cap <<= 1;
				swap_min_index = realloc(swap_min_index, swap_pair_cap * sizeof(uint64_t));
				swap_max_index = realloc(swap_max_index, swap_pair_cap * sizeof(uint64_t));
			}
			swap_min_index[swap_pair_size] = min_empty;
			swap_max_index[swap_pair_size++] = max_full;
		}
		// figure out partitions
		uint64_t dst = min_empty << block_cap_bits;
		uint64_t src = max_full << block_cap_bits;
		// compute thread division of copy
		uint64_t size = block_cap / threads_per_numa;
		uint64_t offset = size * numa_local_id;
		if (numa_local_id + 1 == threads_per_numa)
			size = block_cap - offset;
		// copy data
		copy(&keys[dst + offset], &keys[src + offset], size);
		copy(&rids[dst + offset], &rids[src + offset], size);
		// find next full block
		for (b = max_full - 1 ; b != min_empty ; --b)
			if (local_block_map[b] >= 0) {
				max_full = b;
				break;
			}
		numa_blocks = b + 1;
		if (b == min_empty) break;
	}
	tim = micro_time() - tim;
	a->compact_time = tim;
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
	// update local numa blocks
	assert(numa_blocks <= alloced_blocks);
	if (!numa_local_id) {
		// set total valid blocks
		d->numa_blocks[numa_node] = numa_blocks;
		// update the block map
		for (p = 0 ; p != swap_pair_size ; ++p) {
			min_empty = swap_min_index[p];
			max_full = swap_max_index[p];
			assert(min_empty < max_full);
			assert(max_full >= numa_blocks);
			local_block_map[min_empty] = local_block_map[max_full];
			local_block_map[max_full] = -1;
		}
		free(swap_min_index);
		free(swap_max_index);
		// count blocks of each partition
		uint64_t *count_blocks = calloc(range_partitions, sizeof(uint64_t));
		for (b = 0 ; b != numa_blocks ; ++b) {
			int8_t r = local_block_map[b];
			count_blocks[r]++;
		}
		d->count_blocks[numa_node] = count_blocks;
	}
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
	// generate partition counts and offsets before swaps
	assert(numa_blocks == d->numa_blocks[numa_node]);
	volatile uint64_t *moved = d->moved;
	uint64_t *sizes = malloc(range_partitions * sizeof(uint64_t));
	uint64_t *offsets = malloc(range_partitions * sizeof(uint64_t));
	uint64_t total_blocks = 0;
	for (p = 0 ; p != range_partitions ; ++p) {
		sizes[p] = 0;
		offsets[p] = total_blocks;
		for (n = 0 ; n != numa ; ++n)
			sizes[p] += d->count_blocks[n][p];
		total_blocks += sizes[p];
	}
	// compute how many blocks you must have in each NUMA node
	uint64_t *numa_final_blocks = calloc(numa, sizeof(uint64_t));
	for (p = 0 ; p != range_partitions ; ++p)
		numa_final_blocks[d->numa_dest[p]] += sizes[p];
	// check if space in local numa node is enough
	assert(numa_final_blocks[numa_node] * block_cap +
	       extra_numa_size <= alloced_size);
	// last block on each numa node
	uint64_t *numa_last_offset = malloc(numa * sizeof(uint64_t));
	uint64_t *numa_offsets = malloc(numa * sizeof(uint64_t));
	p = 0;
	for (n = 0 ; n != numa ; ++n) {
		numa_offsets[n] = p;
		p += numa_final_blocks[n];
		numa_last_offset[n] = p - 1;
	}
	assert(p == total_blocks);
#ifdef BG
	if (!numa_local_id) {
		for (b = 0 ; b != numa_blocks ; ++b) {
			int8_t r = local_block_map[b];
			assert(r >= 0);
			check_range(&keys[b << block_cap_bits], block_cap, r, range_delimiter);
		}
		// print total blocks per NUMA
		fprintf(stderr, "Blocks before balance: (%d / %d): %ld\n",
				 numa_node + 1, numa, numa_blocks);
	}
	pthread_barrier_wait(&global_barrier[gb++]);
#endif
	// move data across remote NUMA nodes to achieve correct balancing
	int64_t *diff = malloc(numa * sizeof(int64_t));
	int64_t diff_balance = 0;
	for (n = 0 ; n != numa ; ++n) {
		diff[n] = d->numa_blocks[n] - numa_final_blocks[n];
		diff_balance += diff[n];
	}
	assert(diff_balance == 0);
	// blocks that will be transfered to previous numa nodes
	uint64_t prev_numa_blocks = 0;
	for (n = 0 ; n != numa_node ; ++n)
		// negative => nodes needs more blocks
		if (diff[n] < 0) prev_numa_blocks -= diff[n];
	// bring blocks from remote locations
	tim = micro_time();
	for (b = numa_blocks ; b < numa_final_blocks[numa_node] ; ++b) {
		// source numa node
		for (n = 0 ; n != numa ; ++n)
			if (diff[n] > 0) break;
		assert(n != numa);
		assert(n != numa_node);
		// source block
		q = d->numa_blocks[n] - diff[n]--;
		// check if claimed by previous numa nodes
		if (prev_numa_blocks) {
			prev_numa_blocks--;
			b--;
			continue;
		}
		// part of blocks
		size = block_cap / threads_per_numa;
		offset = size * numa_local_id;
		if (numa_local_id + 1 == threads_per_numa)
			size = block_cap - offset;
		// copy part of block
		uint32_t *keys_src = &d->keys[n][q << block_cap_bits];
		uint32_t *rids_src = &d->rids[n][q << block_cap_bits];
		uint32_t *keys_dst = &keys[b << block_cap_bits];
		uint32_t *rids_dst = &rids[b << block_cap_bits];
		copy(&keys_dst[offset], &keys_src[offset], size);
		copy(&rids_dst[offset], &rids_src[offset], size);
		// set blockmap masks
		if (!numa_local_id) {
			local_block_map[b] = block_map[n][q];
			block_map[n][q] = -1;
		}
	}
	tim = micro_time() - tim;
	a->balance_time = tim;
	numa_blocks = numa_final_blocks[numa_node];
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
#ifdef BG
	if (!numa_local_id) {
		for (b = 0 ; b != numa_blocks ; ++b) {
			int8_t r = local_block_map[b];
			assert(r >= 0);
			check_range(&keys[b << block_cap_bits], block_cap, r, range_delimiter);
		}
		// print total blocks per NUMA
		fprintf(stderr, "Blocks after balance: (%d / %d): %ld\n",
				 numa_node + 1, numa, numa_blocks);
	}
	pthread_barrier_wait(&global_barrier[gb++]);
#endif
	free(numa_final_blocks);
	// store open slots of last blocks
	uint64_t *open_block = malloc(range_partitions * sizeof(uint64_t));
	// allocate buffers (previous extra used for last item processing)
	uint32_t *keys_buf = malloc(block_cap * sizeof(uint32_t));
	uint32_t *rids_buf = malloc(block_cap * sizeof(uint32_t));
	// start moving data in cycles using XADD between blocks
	uint64_t last = p = 0;
	tim = micro_time();
	while (p != range_partitions) {
		// get a new cycle element
		b = __sync_fetch_and_add(&moved[p << 4], 1);
		if (b >= sizes[p]) {
			p++;
			continue;
		}
		b += offsets[p];
		uint64_t cycle_block = b;
		// find which numa node this block belongs to
		n = binary_search_64(numa_last_offset, numa, b);
		b -= numa_offsets[n];
		// point to first block
		uint32_t *cycle_key = &d->keys[n][b << block_cap_bits];
		uint32_t *cycle_rid = &d->rids[n][b << block_cap_bits];
		// find the destination of the block
		uint64_t h = binary_search_32(range_delimiter, range_partitions, cycle_key[0]);
//		uint64_t h = block_map[n][b];
		// if same as source skip it
		if (h == p) continue;
#ifdef BG
		check_range(cycle_key, block_cap, h, range_delimiter);
#endif
		// load the block in the buffer (and cache the buffer)
		block_store(keys_buf, cycle_key, block_cap);
		block_store(rids_buf, cycle_rid, block_cap);
		// cycle until you go back to same partition
		int8_t valid_cycle = 1;
		do {
			// fetch new pointer from destination
			b = __sync_fetch_and_add(&moved[h << 4], 1);
			valid_cycle = b < sizes[h];
			b += offsets[h];
			if (!valid_cycle) break;
			// find which numa node this block belongs to
			n = binary_search_64(numa_last_offset, numa, b);
			b -= numa_offsets[n];
			// point to block
			uint32_t *key = &d->keys[n][b << block_cap_bits];
			uint32_t *rid = &d->rids[n][b << block_cap_bits];
			// find the destination of the block
			h = binary_search_32(range_delimiter, range_partitions, key[0]);
		//	h = block_map[n][b];
#ifdef BG
			check_range(key, block_cap, h, range_delimiter);
#endif
			// swap items with buffer (stream to memory)
			block_swap(key, keys_buf, block_cap);
			block_swap(rid, rids_buf, block_cap);
		// if destination is same as source stop
		} while (h != p);
		if (valid_cycle) {
			// copy from buffer to block that started the cycle
			block_stream(cycle_key, keys_buf, block_cap);
			block_stream(cycle_rid, rids_buf, block_cap);
		} else {
			// destination is private space
			uint32_t *key = &keys_space[last << block_cap_bits];
			uint32_t *rid = &rids_space[last << block_cap_bits];
			open_block[last++] = cycle_block;
			// stream out block to private space
			block_stream(key, keys_buf, block_cap);
			block_stream(rid, rids_buf, block_cap);
#ifdef BG
			check_range(key, block_cap, h, range_delimiter);
#endif
		}
	}
	tim = micro_time() - tim;
	a->block_swap_online_time = tim;
	free(keys_buf);
	free(rids_buf);
	free(numa_last_offset);
	// save info on last copied items
	d->open[id] = last;
	d->open_block[id] = open_block;
	d->keys_space[id] = keys_space;
	d->rids_space[id] = rids_space;
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
	// space to store pointers of last blocks
	uint32_t **last_keys = malloc(threads * sizeof(uint32_t*));
	uint32_t **last_rids = malloc(threads * sizeof(uint32_t*));
	// convert the offsets to point on last block of each partition
	uint64_t *last_offset = malloc(range_partitions * sizeof(uint64_t));
	for (p = 0 ; p != range_partitions ; ++p)
		last_offset[p] = offsets[p] + sizes[p] - 1;
	tim = micro_time();
	for (;;) {
		// skip other partitition data
		p = __sync_fetch_and_add(d->ranges_closed, 1);
		if (p >= range_partitions) break;
		// numa location of partition
		n = d->numa_dest[p];
		// blocks to copy
		last = 0;
		for (t = 0 ; t != threads ; ++t)
			for (l = 0 ; l != d->open[t] ; ++l) {
				// read first key from block and find destination
				uint32_t key = d->keys_space[t][l << block_cap_bits];
				if (binary_search_32(range_delimiter, range_partitions, key) != p)
					continue;
				// save pointers of block
				last_keys[last] = &d->keys_space[t][l << block_cap_bits];
				last_rids[last++] = &d->rids_space[t][l << block_cap_bits];
			}
		if (last == 0) continue;
		// search destinations for last blocks of current partition
		for (t = 0 ; t != threads ; ++t)
			for (l = 0 ; l != d->open[t] ; ++l) {
				// find what partition the block location belongs to
				b = d->open_block[t][l];
				if (p != binary_search_64(last_offset, range_partitions, b))
					continue;
				// block also belong to same numa node
				assert(b >= numa_offsets[n]);
				b -= numa_offsets[n];
				// copy data from private space to actual block locations
				uint32_t *key = &d->keys[n][b << block_cap_bits];
				uint32_t *rid = &d->rids[n][b << block_cap_bits];
				block_stream(key, last_keys[--last], block_cap);
				block_stream(rid, last_rids[last], block_cap);
			}
		// all items were copied to destinations
		assert(last == 0);
	}
	tim = micro_time() - tim;
	a->block_swap_offline_time = tim;
	free(last_offset);
	free(last_keys);
	free(last_rids);
	// synchronize all threads
	pthread_barrier_wait(&global_barrier[gb++]);
	free(keys_space);
	free(rids_space);
	free(numa_offsets);
	free(offsets);
	// find partitions of this numa node
	a->inject_time = 0;
	if (!numa_local_id) {
		for (p = 0 ; d->numa_dest[p] != numa_node ; ++p);
		uint64_t p_from = p;
		for (; d->numa_dest[p] == numa_node ; ++p);
		uint64_t p_to = p;
		uint64_t max_size = d->size[numa_node] * d->fudge;
		tim = micro_time();
		numa_size = inject(d->keys[numa_node], d->rids[numa_node],
				   &sizes[p_from], p_to - p_from, block_cap,
				   &d->half_block_keys[p_from],
				   &d->half_block_rids[p_from],
				   &d->half_block_size[p_from]);
		tim = micro_time() - tim;
		a->inject_time = tim;
		assert(numa_size <= alloced_size);
		d->size[numa_node] = numa_size;
		for (p = p_from ; p != p_to ; ++p) {
			free(d->half_block_keys[p]);
			free(d->half_block_rids[p]);
		}
#ifdef BG
		uint32_t *key = d->keys[numa_node];
		for (p = p_from ; p != p_to ; ++p) {
			uint64_t size = block_cap * sizes[p] + d->half_block_size[p];
			check_range(key, size, p, range_delimiter);
			uint32_t high_bits = key[0] >> 26;
			for (q = 0 ; q != size ; ++q)
				assert((key[q] >> 26) == high_bits);
			key += size;
		}
		fprintf(stderr, "Partitioning for %d / %d checked!\n",
				 numa_node + 1, numa);
#endif
	}
	// synchronize local threads
	pthread_barrier_wait(&local_barrier[lb++]);
	// find thread local starting partition
	int rid = numa_node * threads_per_numa + numa_local_id;
	uint32_t min_delim = !rid ? 0 : thread_delimiter[rid - 1] + 1;
	uint32_t max_delim = thread_delimiter[rid];
	if (max_delim <= min_delim) {
		a->local_sort_tuples = 0;
		a->local_sort_time = 0;
		free(sizes);
		assert(gb < 48 && lb < 32);
		pthread_exit(NULL);
	}
	for (p = 0 ; range_delimiter[p] < min_delim ; ++p);
	uint64_t p_from = p;
	for (; range_delimiter[p] != max_delim ; ++p);
	uint64_t p_to = p + 1;
	// do local MSB radixsort of partitions
	tim = micro_time();
	// point to thread local partitions
	keys = d->keys[numa_node];
	rids = d->rids[numa_node];
	// skip previous numa nodes
	for (p = 0 ; d->numa_dest[p] != numa_node ; ++p);
	assert(p <= p_from);
	// skip previous parts
	for (; p != p_from ; ++p) {
		uint64_t size = sizes[p] * block_cap + d->half_block_size[p];
		keys += size;
		rids += size;
	}
	// local sort parts
	uint64_t local_tuples = 0;
	int8_t radix_bits[8];
	int8_t buffered[8];
	uint64_t *hist[8], *offs[8];
	int8_t allocated_bits[8];
	for (i = 0 ; i != 8 ; ++i) {
		allocated_bits[i] = 0;
		hist[i] = offs[i] = NULL;
	}
	for (; p != p_to ; ++p) {
		uint64_t size = sizes[p] * block_cap + d->half_block_size[p];
		if (size == 0) continue;
		i = schedule_passes(size, 26, radix_bits, buffered);
		for (j = 0 ; j != i ; ++j)
			if (radix_bits[j] > allocated_bits[j]) {
				hist[j] = realloc(hist[j], (1 << radix_bits[j]) * sizeof(uint64_t));
				offs[j] = realloc(offs[j], (1 << radix_bits[j]) * sizeof(uint64_t));
				allocated_bits[j] = radix_bits[j];
			}
		while (i--) radix_bits[i] += radix_bits[i + 1];
		local_radixsort(keys, rids, size, radix_bits, buffered, 0, hist, offs);
		keys += size;
		rids += size;
		local_tuples += size;
	}
	for (i = 0 ; i != 8 ; ++i) {
		free(hist[i]);
		free(offs[i]);
	}
	tim = micro_time() - tim;
	a->local_sort_time = tim;
	a->local_sort_tuples = local_tuples;
	free(sizes);
	assert(gb < 48 && lb < 32);
	pthread_exit(NULL);
}

void sort(uint32_t **keys, uint32_t **rids, uint64_t *size,
          int threads, int numa, double fudge,
          char **description, uint64_t *times)
{
	int i, j, t, n;
	assert(threads == 64);
	// global 6 bits + 64 range partitions (local 26 bits)
	uint64_t p, range_partitions = 128;
	// check aligned input
	for (i = 0 ; i != numa ; ++i) {
		assert((15 & (uint64_t) keys[i]) == 0);
		assert((15 & (uint64_t) rids[i]) == 0);
	}
	// total tuples
	uint64_t total_size = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size += size[n];
	// initialize global barriers
	int local_barriers = 32;
	int global_barriers = 48;
	pthread_barrier_t sample_barrier;
	pthread_barrier_t *global_barrier = malloc(global_barriers * sizeof(pthread_barrier_t));
	pthread_barrier_t **local_barrier = malloc(numa * sizeof(pthread_barrier_t*));
	pthread_barrier_init(&sample_barrier, NULL, threads + 1);
	for (t = 0 ; t != global_barriers ; ++t)
		pthread_barrier_init(&global_barrier[t], NULL, threads);
	// initialize local barriers
	int threads_per_numa = threads / numa;
	for (n = 0 ; n != numa ; ++n) {
		local_barrier[n] = malloc(local_barriers * sizeof(pthread_barrier_t));
		for (t = 0 ; t != local_barriers ; ++t)
			pthread_barrier_init(&local_barrier[n][t], NULL, threads_per_numa);
	}
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	thread_data_t *data = malloc(threads * sizeof(thread_data_t));
	// universal meta data
	global_data_t global;
	global.numa = numa;
	global.threads = threads;
	global.max_numa = numa_max_node() + 1;
	global.max_threads = hardware_threads();
	global.keys = keys;
	global.rids = rids;
	global.size = size;
	global.ranges_compacted = calloc(1, sizeof(uint64_t));
	global.ranges_closed = calloc(1, sizeof(uint64_t));
	global.block_cap = 4096;
	global.fudge = fudge;
	global.global_barrier = global_barrier;
	global.local_barrier = local_barrier;
	global.sample_barrier = &sample_barrier;
	// allocate the sample
	global.sample_size = 0.005 * total_size;
	global.sample_size &= ~15;
	if (global.sample_size > 500000)
		global.sample_size = 500000;
	global.sample	  = numa_alloc_interleaved(global.sample_size * sizeof(uint32_t));
	global.sample_buf = numa_alloc_interleaved(global.sample_size * sizeof(uint32_t));
	global.sample_hist = malloc(threads * sizeof(uint64_t*));
	for (t = 0 ; t != threads ; ++t)
		global.sample_hist[t] = malloc(256 * sizeof(uint64_t));
	// counts
	global.block_map = malloc(numa * sizeof(uint8_t*));
	global.numa_blocks = malloc(numa * sizeof(uint64_t*));
	global.count_blocks = malloc(numa * sizeof(uint64_t*));
	global.more_blocks = calloc(numa, sizeof(uint64_t));
	global.max_block_index = malloc(threads * sizeof(uint64_t*));
	global.thread_total_blocks = malloc(numa * sizeof(uint64_t*));
	global.thread_unused_blocks = malloc(numa * sizeof(uint64_t*));
	global.thread_open_block_index = malloc(numa * sizeof(uint64_t**));
	global.thread_open_block_size = malloc(numa * sizeof(uint64_t**));
	for (n = 0 ; n != numa ; ++n) {
		global.thread_total_blocks[n] = malloc(threads_per_numa * sizeof(uint64_t));
		global.thread_unused_blocks[n] = malloc(threads_per_numa * sizeof(uint64_t));
		global.thread_open_block_index[n] = malloc(threads_per_numa * sizeof(uint64_t*));
		global.thread_open_block_size[n] = malloc(threads_per_numa * sizeof(uint64_t*));
	}
	global.keys_space = malloc(threads * sizeof(uint32_t*));
	global.rids_space = malloc(threads * sizeof(uint32_t*));
	global.moved = calloc(range_partitions << 4, sizeof(uint64_t));
	global.offline_up = calloc(range_partitions, sizeof(uint64_t));
	global.offline_down = calloc(range_partitions, sizeof(uint64_t));
	global.first_keys = malloc(range_partitions * sizeof(uint32_t**));
	global.first_rids = malloc(range_partitions * sizeof(uint32_t**));
	global.first_size = malloc(range_partitions * sizeof(uint64_t*));
	global.half_block_keys = calloc(range_partitions, sizeof(uint32_t*));
	global.half_block_rids = calloc(range_partitions, sizeof(uint32_t*));
	global.half_block_size = calloc(range_partitions, sizeof(uint64_t));
	for (p = 0 ; p != range_partitions ; ++p) {
		global.first_keys[p] = malloc(threads * sizeof(uint32_t*));
		global.first_rids[p] = malloc(threads * sizeof(uint32_t*));
		global.first_size[p] = malloc(threads * sizeof(uint64_t));
	}
	global.open = malloc(threads * sizeof(uint64_t));
	global.open_block = malloc(threads * sizeof(uint64_t*));
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	// spawn threads
	for (t = 0 ; t != threads ; ++t) {
		data[t].id = t;
		data[t].global = &global;
		pthread_create(&id[t], NULL, sort_thread, (void*) &data[t]);
	}
	// free sample data
	pthread_barrier_wait(&sample_barrier);
	pthread_barrier_destroy(&sample_barrier);
	numa_free(global.sample,     global.sample_size * sizeof(uint32_t));
	numa_free(global.sample_buf, global.sample_size * sizeof(uint32_t));
	// join threads
	for (t = 0 ; t != threads ; ++t)
		pthread_join(id[t], NULL);
	// check total size
	uint64_t total_size_after = 0;
	for (n = 0 ; n != numa ; ++n)
		total_size_after += size[n];
	assert(total_size_after == total_size);
	// measure times
	uint64_t st = 0, ptf = 0, pbt = 0, cm = 0, cp = 0;
	uint64_t bt = 0, bon = 0, bof = 0, it = 0, ls = 0;
	for (t = 0 ; t != threads ; ++t) {
		st += data[t].sample_time;
		ptf += data[t].partition_first_time;
		pbt += data[t].partition_blocks_time;
		cm += data[t].combine_time;
		cp += data[t].compact_time;
		bt += data[t].balance_time;
		bon += data[t].block_swap_online_time;
		bof += data[t].block_swap_offline_time;
		it += data[t].inject_time;
		ls += data[t].local_sort_time;
	}
//	for (t = 0 ; t != threads ; ++t)
//		fprintf(stderr, "Thread %2d: %.2f%%\n", t,
//			data[t].local_sort_tuples * 100.0 / total_size);
	times[0] = st / threads;  description[0] = "Sample time:	      ";
	times[1] = ptf / threads; description[1] = "Partition (first) time:   ";
	times[2] = pbt / threads; description[2] = "Partition to blocks time: ";
	times[3] = cm / threads;  description[3] = "Combine blocks time:      ";
	times[4] = cp / threads;  description[4] = "Compact blocks time:      ";
	times[5] = bt / threads;  description[5] = "Balance blocks time:      ";
	times[6] = bon / threads; description[6] = "Swap blocks online time:  ";
	times[7] = bof / threads; description[7] = "Swap blocks offline time: ";
	times[8] = it / numa;	  description[8] = "Injection of data time:   ";
	times[9] = ls / threads;  description[9] = "Local radixsort time:     ";
	description[10] = NULL;
	// destroy barriers
	for (t = 0 ; t != global_barriers ; ++t)
		pthread_barrier_destroy(&global_barrier[t]);
	for (n = 0 ; n != numa ; ++n) {
		for (t = 0 ; t != local_barriers ; ++t)
			pthread_barrier_destroy(&local_barrier[n][t]);
		free(local_barrier[n]);
	}
	free(global_barrier);
	free(local_barrier);
	// release memory
	numa_free(global.sample,     global.sample_size * sizeof(uint32_t));
	numa_free(global.sample_buf, global.sample_size * sizeof(uint32_t));
	free(id);
	free(global.numa_node);
	free(global.cpu);
	free(data);
}

void *check_thread(void *arg)
{
	thread_data_t *a = (thread_data_t*) arg;
	global_data_t *d = a->global;
	int i, id = a->id;
	int numa = d->numa;
	int numa_node = d->numa_node[id];
	int threads = d->threads;
	int threads_per_numa = threads / numa;
	// id in local numa threads
	int numa_local_id = 0;
	for (i = 0 ; i != id ; ++i)
		if (d->numa_node[i] == numa_node)
			numa_local_id++;
	// compute checksum
	uint64_t numa_size = d->size[numa_node];
	uint64_t size = numa_size / threads_per_numa;
	uint64_t offset = size * numa_local_id;
	if (numa_local_id + 1 == threads_per_numa)
		size = numa_size - size * numa_local_id;
	uint32_t *keys = &d->keys[numa_node][offset];
	uint32_t *rids = NULL;
	if (d->rids != NULL)
		rids =	&d->rids[numa_node][offset];
	uint32_t *keys_end = &keys[size];
	uint64_t sum = 0;
	uint32_t pkey = 0;
	while (keys != keys_end) {
		uint32_t key = *keys++;
		if (rids) assert(key == *rids++);
		assert(key >= pkey);
		sum += key;
		pkey = key;
	}
	a->checksum = sum;
	pthread_exit(NULL);
}

uint64_t check(uint32_t **keys, uint32_t **rids, uint64_t *size, int numa, int same)
{
	int max_threads = hardware_threads();
	int n, t, threads = 0;
	for (t = 0 ; t != max_threads ; ++t)
		if (numa_node_of_cpu(t) < numa)
			threads++;
	global_data_t global;
	global.threads = threads;
	global.numa = numa;
	global.keys = keys;
	global.rids = same ? rids : NULL;
	global.size = size;
	global.cpu = malloc(threads * sizeof(int));
	global.numa_node = malloc(threads * sizeof(int));
	schedule_threads(global.cpu, global.numa_node, threads, numa);
	thread_data_t *data = malloc(threads * sizeof(thread_data_t));
	pthread_t *id = malloc(threads * sizeof(pthread_t));
	for (t = 0 ; t != threads ; ++t) {
		data[t].id = t;
		data[t].global = &global;
		pthread_create(&id[t], NULL, check_thread, (void*) &data[t]);
	}
	for (n = 1 ; n != numa ; ++n)
		assert(keys[n][0] >= keys[n - 1][size[n - 1] - 1]);
	uint64_t checksum = 0;
	for (t = 0 ; t != threads ; ++t) {
		pthread_join(id[t], NULL);
		checksum += data[t].checksum;
	}
	free(global.numa_node);
	free(global.cpu);
	free(data);
	free(id);
	return checksum;
}

uint64_t read_from_file(uint32_t **keys, uint64_t *size, int numa, const char *name)
{
	FILE *fp = fopen(name, "r");
	assert(fp != NULL);
	fseek(fp, 0, SEEK_END);
	uint64_t total_size = 0; int n, c;
	for (n = 0 ; n != numa ; ++n)
		total_size += size[n];
	assert(total_size <= ftell(fp));
	fseek(fp, 0, SEEK_SET);
	assert(ftell(fp) == 0);
	uint64_t checksum = 0;
	uint64_t p, perc = total_size / 100;;
	uint64_t q = 1, done = 0;
	for (n = 0 ; n != numa ; ++n) {
		memory_bind(n);
		for (c = 0 ; numa_node_of_cpu(c) != n ; ++c);
		cpu_bind(c);
		uint64_t rem_size = size[n];
		uint32_t *key = keys[n];
		while (rem_size) {
			uint64_t size = 4096;
			if (size > rem_size) size = rem_size;
			size = fread(key, sizeof(uint32_t), size, fp);
			assert(size > 0);
			for (p = 0 ; p != size ; ++p)
				checksum += key[p];
			rem_size -= size;
			key += size;
			done += size;
			if (done > (perc * q))
				fprintf(stderr, "Finished %ld%%\n", q++);
		}
	}
	p = ftell(fp);
	assert(p == total_size * sizeof(uint32_t));
	fclose(fp);
	return checksum;
}

int main(int argc, char **argv)
{
	uint64_t tuples = argc > 1 ? atoi(argv[1]) : 1000;
	int threads = argc > 2 ? atoi(argv[2]) : 64;
	int numa = argc > 3 ? atoi(argv[3]) : numa_max_node() + 1;
	tuples *= (uint64_t) 1000000;
	char *name = NULL;
	double theta = 0.0;
	if (argc > 4)
		if (isdigit(argv[4][0]))
			theta = atof(argv[4]);
		else {
			name = argv[4];
			FILE *fp = fopen(name, "r");
			if (fp == NULL) perror("");
			fclose(fp);
		}
	double fudge = theta != 0.0 ? 2.0 : 1.2;
	assert(numa > 0 && threads >= numa && threads % numa == 0);
	assert(threads == 64);
	uint64_t tuples_per_numa = tuples / numa;
	uint64_t capacity_per_numa = tuples_per_numa * fudge;
	uint32_t *keys[numa], *rids[numa];
	uint32_t *keys_buf[numa], *rids_buf[numa];
	uint64_t size[numa], cap[numa];
	uint32_t seed = micro_time();
	int i, same_key_payload = 1;
	srand(0);
	fprintf(stderr, "Tuples: %.2f mil. (%.1f GB)\n", tuples / 1000000.0,
	                (tuples * 8.0) / (1024 * 1024 * 1024));
	fprintf(stderr, "NUMA nodes: %d\n", numa);
	fprintf(stderr, "Hardware threads: %d (%d per NUMA)\n", threads, threads / numa);
	fprintf(stderr, "Threads: %d (%d per NUMA)\n", threads, threads / numa);
	for (i = 0 ; i != numa ; ++i) {
		size[i] = tuples_per_numa;
		cap[i] = size[i] * fudge;
	}
	// initialize space
	uint64_t sum_k, sum_v, checksum;
	uint64_t t = micro_time();
	srand(0);
	if (argc <= 4) {
		sum_k = init_32(keys, size, cap, threads, numa, 32, 0.0, 0, 0);
		srand(0);
		sum_v = init_32(rids, size, cap, threads, numa, 32, 0.0, 0, 0);
		assert(sum_k == sum_v);
	} else {
		uint64_t ranks[33];
		init_32(keys, size, cap, threads, numa, 0, 0.0, 0, 0);
		if (name != NULL) {
			fprintf(stderr, "Opening file: %s\n", name);
			sum_k = read_from_file(keys, size, numa, name);
		} else {
			uint32_t *values = NULL;
			uint64_t size_32_bit = ((uint64_t) 1) << 32;
			fprintf(stderr, "Generating all 32-bit items\n");
			sum_k = init_32(&values, &size_32_bit, &size_32_bit, threads, 1, -1, 0.0, 0, 1);
			assert(sum_k == size_32_bit * (size_32_bit - 1) / 2);
			fprintf(stderr, "Shuffling all 32-bit items\n");
			shuffle_32(values, size_32_bit);
			fprintf(stderr, "Generating Zipf with theta = %.2f\n", theta);
			sum_k = zipf_32(keys, size, values, numa, theta, ranks);
			numa_free(values, size_32_bit * sizeof(uint32_t));
		}
		same_key_payload = 0;
		sum_v = init_32(rids, size, cap, threads, numa, 32, 0.0, 0, 0);
	}
	t = micro_time() - t;
	fprintf(stderr, "Generation time: %ld us\n", t);
	fprintf(stderr, "Generation rate: %.1f mrps\n", tuples * 1.0 / t);
	fprintf(stderr, "Seed: %u\n", seed);
	// sort info
	// Use for Energy cost measure
	long long * values = NULL;
	int EventSet = RAPL_init(&values);
	int retval = PAPI_start( EventSet );
	if(retval != PAPI_OK || values == NULL) {
		fprintf(stderr, "PAPI_start failed!\n");
		exit(1);
	}

	uint64_t times[12];
	char *desc[12];
	// call parallel sort
	t = micro_time();
	sort(keys, rids, size, threads, numa, fudge, desc, times);
	t = micro_time() - t;
	// print sort times
	retval = PAPI_stop(EventSet, values);
	if(retval != PAPI_OK || values == NULL) {
		fprintf(stderr, "PAPI_stop failed!\n");
		exit(1);
	}
	struct Energy energy = energy_cost(1, values);
	PAPI_cleanup_eventset( EventSet );
	PAPI_destroy_eventset( EventSet );
	fprintf(stderr, "Sort time: %ld us\n", t);
	
	FILE * output = fopen("msb_32.txt", "a");
	fprintf(output, "%f\t%f\t%f\t%f\n", energy.total_package, energy.total_dram, energy.total_energy, (double)t/1e6);
	fclose(output);

	double gigs = (tuples * 8.0) / (1024 * 1024 * 1024);
	fprintf(stderr, "Sort rate: %.1f mrps (%.2f GB / sec)\n", tuples * 1.0 / t, (gigs * 1000000) / t);
	// compute total time
	uint64_t total_time = 0;
	for (i = 0 ; desc[i] != NULL ; ++i)
		total_time += times[i];
	// show part times
	for (i = 0 ; desc[i] != NULL ; ++i)
		fprintf(stderr, "%s %10ld us (%5.2f%%)\n", desc[i],
				 times[i], times[i] * 100.0 / total_time);
	fprintf(stderr, "Noise time loss: %.2f%%\n", t * 100.0 / total_time - 100);
	// show allocation per NUMA node
	for (i = 0 ; i != numa ; ++i)
		fprintf(stderr, "Node %d:%6.2f%%\n", i, size[i] * 100.0 / tuples);
	// check sort order and sum
	checksum = check(keys, rids, size, numa, same_key_payload);
	assert(checksum == sum_k);
	fprintf(stderr, "Checksum: %lu\n", checksum);
	// free sort data
	for (i = 0 ; i != numa ; ++i) {
		free(keys[i]);
		free(rids[i]);
	}
	printf("%.1f mrps (%.2f GB / sec)\n",
		tuples * 1.0 / t, (gigs * 1000000) / t);
	return EXIT_SUCCESS;
}
