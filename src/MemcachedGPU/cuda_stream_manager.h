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
 * cuda_stream_manager.h
 */

#ifndef CUDA_MANAGER_H_
#define CUDA_MANAGER_H_

// TODO: Clean up unused includes from testing 

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <pthread.h>
#include <assert.h>
#include <map>
#include <queue>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <vector>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <cstring>
#include <string>

#include <fstream>
#include <sstream>

#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/in_systm.h>
#include <netinet/ip6.h>
#include <net/ethernet.h>     /* the L2 protocols */
#include <sys/time.h>
#include <time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <linux/if_packet.h>



extern "C" {
#include "gpu_common.h"
#include "memcached.h"
}

#include "cuda_context_manager.h"

/* PF_RING */
extern "C" {
#include "pfring.h"
#include "pfring_zc.h"
}



// CUDA utilities and system includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>


#include <cstddef>


// Uncomment this to do the network only test. This does all networking without any Memcached.
//#define NETWORK_ONLY_TEST

// Uncomment this to add the 8 Byte memcached header to the response packet. Client uses this for latency measurements
//#define LATENCY_MEASURE

#ifndef DO_GNOM_TX // If not doing GNoM TX, do PF_RING TX
#define USE_PF_RING
#endif

#define MAX_PF_RING_TX  8
#define MAX_PF_RING_RX  8

/************ NGD ************/
// Currently the NGD framework requires constant sized
// requests to work. Each packet is copied into a contiguous
// buffer at PKT_STRIDE offsets.

// TODO: Add header to buffer to specify packet locations in memory.

#define PKT_STRIDE 72       // 16 Byte key
//#define PKT_STRIDE 88     // 32 Byte key
//#define PKT_STRIDE 120    // 64 Byte key
//#define PKT_STRIDE 184    // 128 Byte key
/****************************/

#define BUFFER_SIZE     NUM_REQUESTS_PER_BATCH*MAX_REQUEST_SIZE
#define MAX_REQUESTS    NUM_C_KERNELS*NUM_REQUESTS_PER_BATCH
#define MAX_NUM_BATCH_INFLIGHT 512


#define ETH_ALEN        6


/***************** GPU_NoM ******************/
// With a single GPU at 10 GbE, a single GNoM-pre thread performed best (throughput + energy-efficiency).
#define NUM_GNOM_PRE_THREADS 1

// 1 puller thread can handle up to ~8MRPS. Passed this, 2 threads are required to reach 13MRPS line rate. 
// 4 threads doesn't really help latency, but increases energy consumption. 
 #define NUM_GNOM_POST_THREADS	4 //2 //1

// Now always MULTI_PF_RING_TX, but just set to 1 for a single TX port/thread
#define MULTI_PF_RING_TX    NUM_GNOM_POST_THREADS
#define MULTI_PF_RING_RX    NUM_GNOM_PRE_THREADS


#define BIND_CORE
/*** GPU_NoM ***/

//#define DEBUG

//#define USE_DIRECT_ACCESS_MEM_REQUEST // NOTE: Deprecated as of the mmap implementation. Copy all request batch metadata to GPU memory
#define USE_DIRECT_ACCESS_MEM_RESPONSE // Use direct access memory for the response packets

// Size of response network header + memcached header used in the CUDA kernel
#define PKT_MEMC_HDR_SIZE 56

#define NUM_PF_BUFS 8192 // 4096

typedef struct _gpu_nom_req_{
    int in_use;
    int req_id;
	int queue_id;
	int num_req;
	rel_time_t timestamp;
	int batch_id;
	int queue_ind;
	size_t *req_ptr;
	size_t *res_ptr;
	CUdeviceptr gpu_req_ptr;
	CUdeviceptr gpu_res_ptr;
    //double init_time;
	struct timespec start_time;
    void *extra; // used to point to any additional config-dependent data structures
}gpu_nom_req;


typedef enum _REQ_STATUS_{
    SUCCESS = 0,
    FAILURE,
}STATUS;


struct ip_header {

#if BYTE_ORDER == LITTLE_ENDIAN
  u_int32_t	ihl:4,		/* header length */
    version:4;			/* version */
#else
  u_int32_t	version:4,			/* version */
    ihl:4;		/* header length */
#endif
  u_int8_t	tos;			/* type of service */
  u_int16_t	tot_len;			/* total length */
  u_int16_t	id;			/* identification */
  u_int16_t	frag_off;			/* fragment offset field */
  u_int8_t	ttl;			/* time to live */
  u_int8_t	protocol;			/* protocol */
  u_int16_t	check;			/* checksum */
  u_int32_t saddr;
  u_int32_t daddr;	/* source and dest address */
};

/*
 * Udp protocol header.
 * Per RFC 768, September, 1981.
 */
struct udp_header {
  u_int16_t	source;		/* source port */
  u_int16_t	dest;		/* destination port */
  u_int16_t	len;		/* udp length */
  u_int16_t	check;		/* udp checksum */
};



typedef struct _pkt_info_ {     // Struct storing all the necessary packet info to receive, process, and send
    u_int8_t  smac[ETH_ALEN];  /* MAC src/dst addresses */
    u_int8_t  ip_version;
    u_int8_t  l3_proto, ip_tos; /* Layer 3 protocol/TOS */
    u_int32_t   ip_src, ip_dst;   /* IPv4 src/dst IP addresses */
    u_int16_t l4_src_port, l4_dst_port; /* Layer 4 src/dst ports */
}pkt_info;


typedef struct _req_hdr_{
    unsigned num_req;
    int offset_ptr[NUM_REQUESTS_PER_BATCH];
}req_hdr;

typedef struct _lock_req_{
    pt_req_h m_req;
    int *m_lock;
    unsigned blocking;
}lock_req;

typedef enum _kernel_ret_{
    RX_PTR = 0,
    TX_PTR = 1,
    BATCH_ID = 2,
    QUEUE_IND = 3
}kernel_ret;

/*** Error checking ***/
#define checkGPUNoMErrors(val, err_val)  __checkGPUNoMErrors (val, err_val, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkGPUNoMErrors(int val, int err_val, const char *file, const int line){
    if (unlikely(val == err_val)){
        fprintf(stderr, "checkGPUNoMErrors() error = %04d from file <%s>, line %i.\n",
        		val, file, line);
        exit(EXIT_FAILURE);
    }
}

/* GPUDirect */
/********** SHARED STRUCTURES WITH GPU_KM *****************/
typedef struct _kernel_args_ {
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS m_tokens;
    uint64_t m_addr;	// Physical bus address of the mapped GPU page
    uint64_t m_size;	// Size of buffer within page
}kernel_args;

typedef struct _cpu_kernel_args_ {
    void *user_page_va;
    CUdeviceptr cuda_page_va;
    uint64_t page_size;
}cpu_kernel_args;

/**** [num_buffers | * k_args | k_arg1, k_arg2,..., k_argN] ****/
typedef struct _kernel_args_hdr_ {
	int num_pages;	// Number of pages to map. Total # buffers =
	int num_buffers; // Total number of buffers
    
    union{
        kernel_args *gpu_args;
        cpu_kernel_args *cpu_args;
    }buf_meta_data;
    
	int buffer_type;
}kernel_args_hdr;

typedef struct _index_bounds_{
    int start;
    int end;
}index_bounds;

typedef struct _worker_thread_args_{
	int gpu_km_fp;			// GPU-NoM File pointer
	int tid;				// Thread ID
	CUDAContext *cuda_ctx; 	// Shared CUDA context
	void	*csm;			// Pointer to container cuda_stream_manager

	size_t *mmaped_RX_buffer_pointers; // Buffer containing pointers to GRX
    size_t *mmaped_TX_buffer_pointers; // Buffer containing pointers to GTX

    index_bounds queue;
    index_bounds stream;

}worker_thread_args;

typedef struct _gnom_buf_recycle_info_{
    int batch_id;
    int queue_ind;
}gnom_buf_recycle_info;

typedef struct _bg_thread_args{
    int tid;
    CUDAContext *cuda_ctx;
    void *csm;
    int num_iterations;

}bg_thread_args;

typedef struct _gpu_stream_pair_{
	CUevent *gpu_event;
	gpu_nom_req *gnr;
}gpu_stream_pair;

typedef struct _ngd_req_ {
    void *h_ptr;
    CUdeviceptr d_ptr;
    unsigned n_req;
    unsigned tot_size;

    void *h_res_ptr;
    CUdeviceptr d_res_ptr;

    gpu_nom_req *m_gnom_req;

    bool is_free;
} ngd_req;

/********** SHARED STRUCTURES WITH GPU_KM *****************/

inline void __checkCudaErrors(cudaError err, const char *file, const int line );
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line );

/*************** Main GPU offload Class ***************/
class cuda_stream_manager_base{
    
public:
	volatile static bool stop_stream_manager;

	cuda_stream_manager_base(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr);
    ~cuda_stream_manager_base();
    void stop();

    int set_update_gpu_hashtable(void *req, int req_size, void *res, int res_size, rel_time_t timestamp);
    int set_lock();
    int set_unlock();
    void inc_set_hit();
    void inc_set_miss();
    void inc_set_evict();

    // This function checks all of the timestamps for each puller thread
    // to ensure that the timestamp corresponding to the SET request causing 
    // an eviction is after all GET request batches have completed. This
    // ensures the ordering seen on the GPU is preserved on the CPU
    int poll_get_complete_timestamp(int set_evict_timestamp);

    virtual void stop_gpu_nom() = 0;
    virtual void print_stats();
    
    friend class cuda_stream_manager;
    friend class cuda_stream_manager_ngd;
    
protected:

    CUdeviceptr m_gpu_hashtable_ptr;
    CUdeviceptr m_gpu_lock_ptr;

    int *m_main_mem;        // Main Memcached memory
    int *m_res_mem;         // Response pinned memory
    int *m_req_mem;         // Request pinned memory

    int *m_item_locks;

    // CUDA variables
    CUDAContext     *m_context; // Shared context for all threads
    CUstream        *m_streams;  // CUDA streams
    unsigned        m_num_streams;   // Number of streams

    CUdeviceptr d_main_mem;
    CUdeviceptr d_req_mem;
    CUdeviceptr d_res_mem;

    pthread_mutex_t m_host_mutex; // Locking shared host queues
    pthread_mutex_t m_map_mutex[NUM_C_KERNELS]; // Locking shared host queues

    /********************************/
    // Stats
    unsigned long long num_batches_processed;
    unsigned long long num_memc_hits;
    unsigned long long num_memc_misses;

    double tot_time_for_batches;
    unsigned long long tot_num_batches;

    unsigned long long num_conflict_set_get;
    unsigned long long num_set_hit;
    unsigned long long num_set_miss;
    unsigned long long num_set_evict;
    volatile int m_get_complete_timestamp[NUM_GNOM_POST_THREADS];
    /********************************/

    
    /************************ GPU-NoM ************************/
    std::string m_dev;		// Name of GPU kernel module
    int m_gpu_km_fp; 	// GPU kernel module handle

    kernel_args_hdr m_kernel_arg_hdr;       // RX Structure to interface with GPU_km
    kernel_args_hdr m_tx_kernel_arg_hdr;	// TX Structure to interface with GPU_km
    
    // Pipelined worker threads, splitting tasks
    pthread_t m_gnom_pre_threads[NUM_GNOM_PRE_THREADS];
    pthread_mutex_t m_gnom_pre_mutex; //pusher_launcher_mutex;

    worker_thread_args wta2[NUM_GNOM_PRE_THREADS + NUM_GNOM_POST_THREADS];

    pthread_t m_gnom_post_threads[NUM_GNOM_POST_THREADS];
    pthread_mutex_t m_gnom_post_mutex;


    pthread_mutex_t gnom_recycle_mutex;
    pthread_mutex_t pf_ring_send_mutex;

    void *mmaped_req_ptr;
    void *mmaped_res_ptr;
    int mmaped_req_res_ptr_len;


    // Set requests
    pthread_t m_set_thread;
    static void *set_thread_main(void *arg);


    gpu_nom_req *m_gpu_nom_req;
    
    int current_gpu_nom_req_ind;
    volatile int num_gpu_nom_req_in_flight;


    // Request and Response GPU and CPU buffers
#ifndef SINGLE_GPU_PKT_PTR
    // Note: Don't require host_request_buffers, m_gpu_nom_req takes care of that
    CUdeviceptr gpu_nom_request_buffers[MAX_NUM_BATCH_INFLIGHT];
#endif

    CUdeviceptr *gpu_nom_response_buffers;
    size_t **host_response_buffers;
    

    CUevent *m_gpu_event_queue;


    /***********************************************************************/
    // GNOM: Combined Request and Response pointer buffers.
    // GNoM handles selection of buffers and explicit/implicit data transfers
    // CUdeviceptr gnom_req_res_ptr_buffer[MAX_NUM_BATCH_INFLIGHT];

    /*** Only if doing GNoM TX - not using PF_RING ***/
    
    CUdeviceptr gnom_response_buffers[MAX_NUM_BATCH_INFLIGHT];
    std::map<CUdeviceptr, void *> res_cuda_to_host_map;
    /***********************************************************************/

    typedef std::queue<gpu_stream_pair> gpu_nom_kernel_event_queue;
    gpu_nom_kernel_event_queue *m_k_event_queue[NUM_C_KERNELS];

    // Common GNoM routines
    bool gnom_init_req_res_buffers(int num_buff, int req_size, int res_size);
    bool gnom_launch_pre_post_threads(void *(*pre)(void *), void *(*post)(void *));
    
    bool gnom_init_pf_ring_tx(int cluster_id);
    

    /************************ GNoM-KM ************************/
    bool init_gpu_km();
    bool reg_gpu_km_grxb(kernel_args_hdr *ka);
    bool reg_gpu_km_gtxb(kernel_args_hdr *ka);
    bool signal_gpu_km();
    
    /*********************************************************/

    pthread_mutex_t m_worker_mutex;

    CUstream m_set_stream; // Separate CUDA stream for SET requests
    pthread_mutex_t m_set_mutex; // Locking SET operations
    CUdeviceptr d_set_req;
    CUdeviceptr d_set_res;
    /*********************************************************/

    
    /************************************************************/
    /********************* PF_RING TX ***************************/
    /************************************************************/
    pfring_zc_cluster *m_pf_zc[MULTI_PF_RING_TX];
    pfring_zc_queue *m_pf_zq[MULTI_PF_RING_TX];
    pfring_zc_pkt_buff *m_pf_buffers[MULTI_PF_RING_TX][NUM_PF_BUFS];
    /************************************************************/
    
    /** Debug routines **/
    void print_pkt_hdr(void *hdr);
    int verify_pkt_hdr(void *hdr);

    /*********** Background task ***********/
    pthread_t bg_thread;
    static void *bg_thread_main(void *arg);
    bg_thread_args bg_args;
    pthread_mutex_t bg_thread_mutex;
    /***************************************/

};

// Main GNoM GPUDirect framework
class cuda_stream_manager : public cuda_stream_manager_base {

public:
    cuda_stream_manager(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr);
    ~cuda_stream_manager();
    
    virtual void stop_gpu_nom();
    
private:
    
    // Main GNoM-host threads
    static void *gnom_pre(void *arg);
    static void *gnom_post(void *arg);
    
};

// Non-GPUDirect (ngd) framework
class cuda_stream_manager_ngd : public cuda_stream_manager_base {
    
public:
    cuda_stream_manager_ngd(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr);
    ~cuda_stream_manager_ngd();
    
    virtual void stop_gpu_nom();

private:
    
    std::deque< ngd_req* > m_ngd_free_reqs;

    pthread_mutex_t m_ngd_free_mutex;
    int m_n_batches;

    // GNoM-pre
    static void *gnom_pre(void *arg);
    // GNoM-post
    static void *gnom_post(void *arg);

    bool gnom_ngd_read(ngd_req **rb, int tid);
    bool gnom_ngd_write(ngd_req *rb);


    bool gnom_rx_read_batch(void **host_rx_batch, CUdeviceptr *gpu_rx_batch, int *num_requests);
    bool gnom_rx_recycle(); 

    bool gnom_init_grxb();
    bool gnom_init_pf_ring_rx(int cluster_id);

    /************************************************************/
    /********************* PF_RING RX ***************************/
    /************************************************************/
    pfring_zc_cluster *m_pf_cluster;
    //pfring_zc_cluster *m_pf_zc_rx[MULTI_PF_RING_RX];
    pfring_zc_queue *m_pf_zq_rx[MULTI_PF_RING_RX];
    pfring_zc_pkt_buff *m_pf_buffers_rx[MULTI_PF_RING_RX][NUM_PF_BUFS];

    int m_pf_buf_ind[MULTI_PF_RING_RX];

    pthread_mutex_t m_pfring_rx_mutex;
};

#endif /* CUDA_MANAGER_H_ */











