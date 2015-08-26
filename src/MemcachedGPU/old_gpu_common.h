/*
 * gpu_common.h
 *
 *  Created on: 2012-12-20
 *      Author: tayler
 */

#ifndef GPU_COMMON_H_
#define GPU_COMMON_H_

#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <vector_types.h>

/* GNoM-KM */
#include "../GNoM_km/gpu_km_shared.h"


//#define DO_GNOM_TX    // Enable if GNoM handles TX. Disable if PF_RING handles TX

////#define SINGLE_GPU_PKT_PTR
////#define SINGLE_GPU_BUFER

#define USE_KEY_HASH
#define KEY_HASH_MASK   0x0000000FF
#define SET_ASSOC_SIZE      16

#define NETWORK_HDR_SIZE 42
#define RESPONSE_HDR_SIZE 48    // Hdr + "VALUE "
#define RESPONSE_HDR_STRIDE 256 // 42 bytes for header + 6 bytes for "VALUE " + key


/*************************************/
#define REQUEST_GROUP_SIZE		128 // Don't change

////#define NUM_REQUESTS_PER_BATCH   512  //256 // TODO: Match with "MAX_REQ_BATCH_SIZE" in gpu_km_shared.h
/*************************************/

#define MAX_REQUEST_SIZE    1024

#define NUM_C_KERNELS  28 //24

//#define CONSTANT_RESPONSE_SIZE
// Needs to be 72 for 13 MRPS at 10Gbps
// Needs to be 80 when doing latency measurements
#define RESPONSE_SIZE	72 //80

#define MAX_KEY_SIZE 140 // 250

/********** GPU-NoM ***********/
// Registering GPU resident CUDA buffers

////#define GPU_REG_SINGLE_BUFFER_CMD	96
////#define GPU_REG_MULT_BUFFER_CMD	97

////#define SIGNAL_NIC		98
////#define SHUTDOWN_NIC	99
////#define STOP_SYSTEM     100
////#define GNOM_TX_SEND    101
////#define GNOM_MAP_USER_PAGE  102

// Registering CPU resident CUDA buffers
////#define GNOM_REG_MULT_CPU_BUFFERS 110
////#define GNOM_UNREG_MULT_CPU_BUFFERS 111


////#define TEST_MAP_SINGLE_PAGE 200
////#define TEST_UNMAP_SINGLE_PAGE 201
////#define TEST_SEND_SINGLE_PACKET 202
////#define TEST_CHECK_SEND_COMPLETE 203





// NIC page, buffer, ring defines
/*********** RX ***********/
#define NUM_GPU_RINGS			1

// 112640 = 220MB of pinned GPU memory. This is the maximum amount on Tesla K20c driver 340.65
#define NUM_GPU_BUF_PER_RING	112640 //1536 //512
//#define NUM_GPU_BUF_PER_RING    3072


#define RING_BUF_MULTIPLIER		1 //32
#define RX_PAGE_SZ				1024*64



// FIXME:   GPUDirect requires RX_BUFFER_SZ==2048.
//          NGD on Maxwell requires 1024
#define RX_BUFFER_SZ 			2048
//#define RX_BUFFER_SZ            1024

#define NUM_GPU_BUFFERS			NUM_GPU_RINGS*NUM_GPU_BUF_PER_RING*RING_BUF_MULTIPLIER
#define NUM_GPU_PAGES			NUM_GPU_BUFFERS / (RX_PAGE_SZ/RX_BUFFER_SZ)

/*********** TX ***********/
#define NUM_GPU_TX_RINGS        1

#define NUM_GPU_TX_BUF_PER_RING 8192 //4096

#define RING_TX_BUF_MULTIPLIER  32

//#define TX_BUFFER_SZ            2048
#define TX_BUFFER_SZ            1024

#define NUM_GPU_TX_BUFFERS      NUM_GPU_TX_RINGS*NUM_GPU_TX_BUF_PER_RING*RING_TX_BUF_MULTIPLIER
#define NUM_GPU_TX_PAGES        NUM_GPU_TX_BUFFERS / (TX_PAGE_SZ/TX_BUFFER_SZ)
/******************************/


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


extern void *cpu_items[1024];



#endif /* GPU_COMMON_H_ */
