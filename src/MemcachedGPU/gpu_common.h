// Copyright (c) 2015, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
 * gpu_common.h
 */

#ifndef GPU_COMMON_H_
#define GPU_COMMON_H_

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <vector_types.h>

// Shared file with GNoM-KM 
// A lot of configuration/test defines are in GNoM_km/gpu_km_shared.h
#include "../GNoM_km/gpu_km_shared.h"


#define USE_KEY_HASH
#define KEY_HASH_MASK   0x0000000FF
#define SET_ASSOC_SIZE      16

#define NETWORK_HDR_SIZE 42
#define RESPONSE_HDR_SIZE 48    // Hdr + "VALUE "
#define RESPONSE_HDR_STRIDE 256 // 42 bytes for header + 6 bytes for "VALUE " + key


#define MAX_REQUEST_SIZE    1024

#define NUM_C_KERNELS  28 //24

//#define CONSTANT_RESPONSE_SIZE
// Needs to be 72 for 13 MRPS at 10Gbps
// Needs to be 80 when doing latency measurements
#define RESPONSE_SIZE	72 //80

#define MAX_KEY_SIZE 140 // 250

typedef struct _pt_req_h{
	size_t req_id;
	size_t app_id;	// 0 = Memcached GET request, 1 = Memcached lock request
	size_t queue_id;
	int *req_ptr;
	size_t req_sz;
	int *res_ptr;
	size_t res_sz;
	size_t time;
    ulong4 hash_mask;
	/*
    size_t time_fw_start;
    size_t time_fw_end;
    size_t time_gpu_start;
    size_t time_gpu_end;
    */
	int (*callback_function)(int req_id);
	void (*ack_function)(int req_id);
}pt_req_h;

typedef struct _gpu_primary_hashtable_{
	void *item_ptr;
	unsigned last_accessed_time;
	unsigned valid;
#ifdef USE_KEY_HASH
	unsigned key_hash; // 8-bit key hash - using 4 bytes to keep everything aligned
#endif
	unsigned key_length;
	unsigned pkt_length;
	char key[MAX_KEY_SIZE]; 
}gpu_primary_hashtable;

typedef struct _gpu_set_req_{
	void *item_ptr;
	unsigned init_hv;
	unsigned key_length;
    unsigned pkt_length;
	unsigned char key[MAX_KEY_SIZE];
}gpu_set_req;

typedef struct _gpu_set_res_{
	int host_signal;
	int is_evicted;
    int is_last_get;
	unsigned evicted_hv;
    unsigned evicted_lru_timestamp;
	void *evicted_ptr;
}gpu_set_res;

extern double TSC_MULT;
extern int timer_init;

void init_timer();
__inline__ uint64_t RDTSC(void);
__inline__ uint64_t RDTSC(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ (
    "       xorl %%eax, %%eax \n"
    "       cpuid"
    ::: "%rax", "%rbx", "%rcx", "rdx");
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}

void *init_pfring();			// PF_RING initialization
void *cpu_get_entry_point();	// Hook into CPU GET handler

void *network_poll();		// Thread that polls network and pushes requests to persistent thread framework
void *GPU_response_poll();	// Thread that polls persistent thread response queue and sends response back to client


#endif /* GPU_COMMON_H_ */
