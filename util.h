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


#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdint.h>


struct Energy{
	double total_energy;
	double total_dram;
	double total_package;
} ;

struct Energy 
energy_cost(int print, long long * values);
uint64_t init_32(uint32_t **data, uint64_t *size,
                 uint64_t *capacity, int threads, int numa, int bits,
                 double hh_percentage, int hh_bits, int mode);

uint64_t init_64(uint64_t **data, uint64_t *size,
                 uint64_t *capacity, int threads, int numa, int bits,
                 double hh_percentage, int hh_bits, int mode);

uint64_t zipf_32(uint32_t **data, uint64_t *size, uint32_t *values,
                 int numa, double theta, uint64_t *log_ranks);

uint64_t zipf_64(uint64_t **data, uint64_t *size, uint32_t *values,
                 int numa, double theta, uint64_t *log_ranks);

void shuffle_32(uint32_t *data, uint64_t size);

void shuffle_64(uint64_t *data, uint64_t size);

#endif
