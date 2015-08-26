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
 * cuda_stream_manager.cpp
 */

#include "cuda_stream_manager.h"
#include <helper_cuda_drvapi.h>

#include <sched.h>
#include <numa.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sys/ucontext.h>
#include <queue>

#define hashsize(n) ((unsigned)1<<(n))
#define hashmask(n) (hashsize(n)-1)

/******************************************/
// Background CUDA kernel experiment.
// Total of 256 blocks need to execute
//      Vary the DIVISOR to select how
//      many blocks execute per kernel.
//      This defines how much of the GPU
//      resources are used by the BG
//      task versus still being allocated
//      to MemcachedGPU
#define BLOCKS  256
#define DIVISOR 1
#define BLOCKS_PER_GRID (BLOCKS/DIVISOR)

//#define BG_TEST
#ifdef BG_TEST
//#define BG_TEST_ONLY
#endif

volatile int background_done_flag = 0;
volatile int background_start_flag = 0;
/*****************************************/

// Global variables
volatile bool cuda_stream_manager_base::stop_stream_manager;

static int global_req_id = 0;
static cuda_stream_manager_base *global_csm = NULL;

double average_dispatch_time = 0.0;
double total_time_per_stream[NUM_C_KERNELS];
unsigned long long count_per_stream[NUM_C_KERNELS];
const char *END = "END\r\n";
const char *NF = "ITEM NOT FOUND\r\n";
const char *VAL = "VALUE ";

unsigned long set_count = 0;

static uint64_t total_num_pkt_processed = 0;

extern "C" int gpu_launch_kernel(int blocks, int threads, size_t stream, int *main_mem,  int *req_mem, int *res_mem, int iterations);

extern "C" { // Main Memcached data structures
    extern item** primary_hashtable;
    extern item** old_hashtable;
    extern bool expanding;
    extern unsigned int expand_bucket;
}

#ifndef likely
#define likely(_x)	__builtin_expect((_x), 1)
#define unlikely(_x)	__builtin_expect((_x), 0)
#endif


///////////////////////////////////
///////// HELPER FUNCTIONS ////////
///////////////////////////////////

void segfault_handler(int signal, siginfo_t *info, void *arg);
void sig_handler(int sig);
int set_thread_affinity(unsigned core);
void print_index(const char *string, int tid, int start, int end);
void inc_wrap_index(int &ind, int start, int end);

int in_cksum(unsigned char *buf, unsigned nbytes, int sum);
static u_int32_t wrapsum (u_int32_t sum);
void test_mapping_cuda_buffer(CUdeviceptr dev_ptr, size_t *host_ptr, int gnom_handle);
int verify_pkt(void *data);
void test_spoof_packet(size_t *data);
///////////////////////////////////
///////////////////////////////////
///////////////////////////////////

using namespace std;


// General purpose registers for debugging without GDB
/* Instruction pointer of faulting instruction */
static const char *gregs[] = {
  "REG_R8", "REG_R9", "REG_R10", "REG_R11", "REG_R12",
  "REG_R13", "REG_R14", "REG_R15", "REG_RDI", "REG_RSI",
  "REG_RBP", "REG_RBX", "REG_RDX", "REG_RAX", "REG_RCX",
  "REG_RSP", "REG_RIP", "REG_EFL", "REG_CSGSFS", "REG_ERR",
  "REG_TRAPNO", "REG_OLDMASK", "REG_CR2"
};




////////////////////////////////////////////////////////
///////////////////////// BASE /////////////////////////
////////////////////////////////////////////////////////


/// Constructor
cuda_stream_manager_base::cuda_stream_manager_base(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr){

    m_context = context;
    m_gpu_hashtable_ptr = gpu_hashtable_ptr;
    m_gpu_lock_ptr = gpu_lock_ptr;

    // Legacy pointers, init to NULL
    m_main_mem = NULL;
    m_res_mem = NULL;
    m_req_mem = NULL;

    m_streams = context->streams;
    m_num_streams = context->n_streams; // NOTE: n_streams+1 total
    m_set_stream = context->streams[m_num_streams-1]; // Actually n_streams+1 streams, not out of bounds.. Probably should just fix this hack.

    stop_stream_manager = false;

    // Basing this off of a 64bit pointer. Verify that this holds on the current system.
    if(sizeof(void *) != sizeof(CUdeviceptr))
        abort();

    // Push context for this thread
    CUresult status = cuCtxPushCurrent(m_context->hcuContext);
    if(status != CUDA_SUCCESS){
        printf("Stream_manager_init: Context push error: %d\n", status);
        exit(1);
    }
    
    
    /******** PREVIOUSLY IN INIT_GPU_KM ********/
    
    /****** Register Segfault Handler ******/
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    
    // Register for ctrl+c / ctrl+z to gracefully exit
    signal(SIGTERM, sig_handler);
    signal(SIGINT,  sig_handler);
    
    /****** Register Segfault Handler ******/
    
    // Initialize stats
    num_memc_hits = 0;
    num_memc_misses = 0;
    num_batches_processed = 0;

    current_gpu_nom_req_ind = 0;
    num_gpu_nom_req_in_flight = 0;

    tot_time_for_batches = 0;
    tot_num_batches = 0;

    num_conflict_set_get = 0;
    num_set_hit = 0;
    num_set_miss = 0;
    num_set_evict = 0;
    
    for(unsigned i=0; i<NUM_GNOM_POST_THREADS; ++i){
        m_get_complete_timestamp[i] = 0;
    }
    
    // Initialize locks
    cout << "Initializing mutexes...";
    pthread_mutex_init(&m_worker_mutex, NULL);
    pthread_mutex_init(&m_gnom_pre_mutex, NULL);
    pthread_mutex_init(&m_gnom_post_mutex, NULL);
    pthread_mutex_init(&gnom_recycle_mutex, NULL); // Mutex for recycling RX buffers with GNoM
    pthread_mutex_init(&pf_ring_send_mutex, NULL); // Mutex for sending packets to PF_RING
    pthread_mutex_init(&bg_thread_mutex, NULL);
    pthread_mutex_init(&m_set_mutex, NULL);
    pthread_mutex_init(&m_host_mutex, NULL);
    for(unsigned i=0; i<NUM_C_KERNELS; ++i){
        pthread_mutex_init(&m_map_mutex[i], NULL);
    }
    cout << "Complete" << endl;
    
    // initialize Memcached SET request structures
    int set_size = 2*sizeof(gpu_set_req) + sizeof(gpu_set_res);

    if(cuMemAlloc(&d_set_req, set_size) != CUDA_SUCCESS)
        printf("**ERROR** Couldn't allocate SET request buffer...\n");

    if(cuMemAlloc(&d_set_res, set_size) != CUDA_SUCCESS)
        printf("**ERROR** Couldn't allocate SET response buffer...\n");


    // Initialize maps
    for(unsigned i=0; i<NUM_C_KERNELS; ++i){
        m_k_event_queue[i] = new gpu_nom_kernel_event_queue;
    }

    /********************************************/

    cuCtxPopCurrent(NULL);
}

/// Destructor
cuda_stream_manager_base::~cuda_stream_manager_base(){
    // Cleanup queues
    cout << "cuda_stream_manager_base destructor: Nothing to do..." << endl;
}

bool cuda_stream_manager_base::gnom_init_req_res_buffers(int num_buff, int req_size, int res_size){
    
    //   Old TODO required for GNoM TX. Working, but not as efficient as it could be - Needs optimizations
    // - Need to double the gpu_nom_request_buffers array and remove the host_response_buffers array.
    //   gpu_nom_request_buffers should hold pointers to both request and response buffers now.
    // - Need to add to the gnom->read() to pull both RX and TX buffer pointers
    // - Need to modify the offsets into the response buffer in the CUDA kernel
    // - Need to remove the PF_RING send and add the gnom->ioctl(..., send) function

    cout << "Allocating Response buffers in GPU memory...";
    
    for(unsigned i=0; i<num_buff; ++i){
        m_gpu_nom_req[i].in_use = 0;
        m_gpu_nom_req[i].extra = NULL;
#ifdef DO_GNOM_TX
        checkCudaErrors(cuMemAlloc_v2(&gnom_response_buffers[i], req_size));
#endif

#ifndef SINGLE_GPU_PKT_PTR
        // Host memory for request buffers already allocated and mapped from Kernel space above
        checkCudaErrors(cuMemAlloc_v2(&gpu_nom_request_buffers[i], req_size)); // GPU buffer for requests
#endif

        // Response buffers
#ifdef USE_DIRECT_ACCESS_MEM_RESPONSE
        checkCudaErrors(cuMemAllocHost((void **)&host_response_buffers[i], res_size)); // Host
        checkCudaErrors(cuMemHostGetDevicePointer_v2(&gpu_nom_response_buffers[i], host_response_buffers[i], 0)); // Device
#else
        host_response_buffers[i] = (size_t *)malloc(res_size);    // Host
        checkCudaErrors(cuMemAlloc(&gpu_nom_response_buffers[i], res_size));    // Device
#endif
    }

    cout << "Complete" << endl;
    return true;
}

bool cuda_stream_manager_base::gnom_launch_pre_post_threads(void *(*pre)(void *), void *(*post)(void *)){

    int ind = 0;
    int queue_range_per_thread = 0;
    int stream_range_per_thread = 0;
    
    cout << "Creating " << NUM_GNOM_PRE_THREADS << " GNoM_pre threads..." << endl;
    
    stream_range_per_thread = (int)(NUM_C_KERNELS / NUM_GNOM_PRE_THREADS);
    queue_range_per_thread = (int)(MAX_NUM_BATCH_INFLIGHT / NUM_GNOM_PRE_THREADS); // Number of gpu_req indexes to look at
    
    for(unsigned i=0; i<NUM_GNOM_PRE_THREADS; ++i, ++ind){
        wta2[ind].gpu_km_fp = m_gpu_km_fp;
        wta2[ind].tid = i;
        wta2[ind].cuda_ctx = m_context;
        wta2[ind].csm = (void *)this;

        wta2[ind].mmaped_RX_buffer_pointers = (size_t *)mmaped_req_ptr;
        wta2[ind].mmaped_TX_buffer_pointers = (size_t *)mmaped_res_ptr;

        wta2[ind].stream.start = i*stream_range_per_thread;
        wta2[ind].stream.end = (i+1)*stream_range_per_thread;

        wta2[ind].queue.start = i*queue_range_per_thread;
        wta2[ind].queue.end = (i+1)*queue_range_per_thread;


        if(pthread_create(&m_gnom_pre_threads[i], NULL, pre, (void *)&wta2[ind])){
            printf("ERROR: Cannot create GNoM_pre...\n");
            abort();
        }
    }
    
    cout << "==Complete==" << endl;
    
    cout << "Creating " << NUM_GNOM_POST_THREADS << " GNoM_post threads..." << endl;
    for(unsigned i=0; i<NUM_GNOM_POST_THREADS; ++i, ++ind){
        wta2[ind].gpu_km_fp = m_gpu_km_fp;
        wta2[ind].tid = i;
        wta2[ind].cuda_ctx = m_context;
        wta2[ind].csm = (void *)this;
        wta2[ind].mmaped_RX_buffer_pointers = (size_t *)mmaped_req_ptr;
        wta2[ind].mmaped_TX_buffer_pointers = (size_t *)mmaped_res_ptr;

        if(pthread_create(&m_gnom_post_threads[i], NULL, post, (void *)&wta2[ind])){
            printf("ERROR: Cannot create GNoM_post...\n");
            abort();
        }

    }
    cout << "==Complete==" << endl;
    
    return true;
}


bool cuda_stream_manager_base::gnom_init_pf_ring_tx(int cluster_id){
    int num_queue_buffers = 32768;

    // Each open a different Queue on the device
    const char *pf_devices[MAX_PF_RING_TX] = {"zc:eth3@0", "zc:eth3@1", "zc:eth3@2", "zc:eth3@3",
        "zc:eth3@4", "zc:eth3@5", "zc:eth3@6", "zc:eth3@7"};

    for(unsigned i=0; i<MULTI_PF_RING_TX; ++i){
        m_pf_zc[i] = pfring_zc_create_cluster(cluster_id + i,
                                              1536,
                                              0,
                                              num_queue_buffers + NUM_PF_BUFS,
                                              0,
                                              NULL);

        if(m_pf_zc[i] == NULL) {
            fprintf(stderr, "pfring_zc_create_cluster %d error [%s] Please check your hugetlb configuration\n", i, strerror(errno));
            return false;
        }
        printf("pfring_zc_create_cluster %d success...\n", i);

        for (unsigned j = 0; j < NUM_PF_BUFS; j++) {
            m_pf_buffers[i][j] = pfring_zc_get_packet_handle(m_pf_zc[i]);
            if (m_pf_buffers[i][j] == NULL) {
                fprintf(stderr, "pfring_zc_get_packet_handle error\n");
                return false;
            }
        }
        printf("pfring_zc_get_packet_handle %d success...\n", i);

        m_pf_zq[i] = pfring_zc_open_device(m_pf_zc[i], pf_devices[i], tx_only, 0);

        if(m_pf_zq[i] == NULL) {
            fprintf(stderr, "pfring_zc_open_device %d error [%s] Please check that %s is up and not already used\n",
                    i, strerror(errno), pf_devices[i]);
            return false;
        }

        fprintf(stderr, "PF_RING %d sending packets through %s\n\n", i, pf_devices[i]);

    }
    
    return true;
}

void *cuda_stream_manager::gnom_pre(void *arg){

    worker_thread_args *m_wta = (worker_thread_args *)arg;
    cuda_stream_manager *csm = (cuda_stream_manager *)m_wta->csm;
    int m_tid = m_wta->tid;
    size_t *RXbp = m_wta->mmaped_RX_buffer_pointers;
    size_t *TXbp = m_wta->mmaped_TX_buffer_pointers;

    int batch_launch_count = 0;

    int start_index = m_wta->queue.start;
    int end_index = m_wta->queue.end;
    int current_index = start_index;

    checkCudaErrors(cuCtxPushCurrent(m_wta->cuda_ctx->hcuContext));

    int *gpu_nom_ret = (int *)malloc(sizeof(int)*5);
    int RX_buf_offset = 0;
    int TX_buf_offset = 0;
    size_t *req_batch;

    gpu_nom_req *gnr;

    CUevent *m_event = NULL;

    int start_stream_index = m_wta->stream.start;
    int end_stream_index = m_wta->stream.end;
    int current_stream_index = start_stream_index;

    int start_event_index = m_wta->queue.start;
    int end_event_index = m_wta->queue.end;
    int current_event_index = start_event_index;

    pthread_mutex_lock( &csm->m_host_mutex );
    print_index((const char*)"Pusher launcher thread queue", m_tid, start_index, end_index);
    print_index((const char*)"Launcher thread queue", m_tid, start_event_index, end_event_index);
    print_index((const char*)"Launcher thread stream", m_tid, start_stream_index, end_stream_index);
    pthread_mutex_unlock( &csm->m_host_mutex );

    /*
     int threadsPerBlock = NUM_REQUESTS_PER_BATCH; // 1 thread per request (TODO: Make this configurable)
     int blocksPerGrid   = (2*NUM_REQUESTS_PER_BATCH)/threadsPerBlock;//1; //(size + threadsPerBlock - 1) / threadsPerBlock;
     */
    int threadsPerBlock = 512;
    int blocksPerGrid = 0;
    if(NUM_REQUESTS_PER_BATCH <  512){
        //    threadsPerBlock = NUM_REQUESTS_PER_BATCH; // Maximum number of threads per block
    }

    blocksPerGrid = NUM_REQUESTS_PER_BATCH / 256; // 256 requests per block, 512 threads per block

    printf("threadsPerBlock=%d, blocksPerGrid=%d\n", threadsPerBlock, blocksPerGrid);
    //assert(blocksPerGrid==1);

    // Kernel Arguments. Fill in NULL arguments at runtime
#ifdef NETWORK_ONLY_TEST
    void *cuda_args[] = { NULL, NULL, NULL, NULL};
#else
    // Last two arguments for DEBUG
    void *cuda_args[] = { NULL, NULL, NULL, (void *)&hashpower,  (void *)&csm->m_gpu_hashtable_ptr, (void *)&csm->m_gpu_lock_ptr, NULL /* , NULL, NULL */};
#endif
    rel_time_t timestamp = 0;

    gpu_stream_pair m_gsp;


    struct timespec start_t, end_t;
#ifdef BIND_CORE
    //set_thread_affinity(m_tid);
    set_thread_affinity(0);
#endif

    unsigned long long num_batches = 0;

    double dispatch_sum = 0.0;
    unsigned long long dispatch_count = 0;
    //clock_gettime(CLOCK_REALTIME, &start_t);
    double dispatch_time = 0.0;
    int first_run = 0;
    do{
        //current_index = csm->current_gpu_nom_req_ind;
        gnr = (gpu_nom_req *)&csm->m_gpu_nom_req[current_index];

        if(gnr->in_use){
            printf("ERROR: Ran out of requests...\n");
            continue;
        }
        gnr->in_use = 1;

#ifndef SINGLE_GPU_PKT_PTR
        gnr->gpu_req_ptr = csm->gpu_nom_request_buffers[current_index];  // GPU req ptr
#endif

#ifndef DO_GNOM_TX
        gnr->gpu_res_ptr = csm->gpu_nom_response_buffers[current_index]; // GPU res ptr
        gnr->res_ptr = csm->host_response_buffers[current_index];       // Host res ptr
#else
        gnr->gpu_res_ptr = csm->gnom_response_buffers[current_index]; // GPU res ptr
#endif


        inc_wrap_index(current_index, start_index, end_index);

        gnr->num_req = read(m_wta->gpu_km_fp, (void *)gpu_nom_ret, 4*sizeof(int)); // Reads batch from GPU-NoM

        //printf("Batch %d launched in stream %d!\n", batch_launch_count++, current_stream_index);

        if(!first_run){
            first_run = 1;
        }else{
            clock_gettime(CLOCK_REALTIME, &end_t);
            dispatch_sum +=  (double)((end_t.tv_sec - start_t.tv_sec) + ((end_t.tv_nsec - start_t.tv_nsec)/1E9));
            dispatch_count++;
        }
        clock_gettime(CLOCK_REALTIME, &start_t);

        if(!gnr->num_req) // No data returned, bail out
            break;

        RX_buf_offset = gpu_nom_ret[RX_PTR]; // First ret val is the buffer offset

#ifdef DO_GNOM_TX
        TX_buf_offset = gpu_nom_ret[TX_PTR];
        if(RX_buf_offset > 511 || TX_buf_offset > 511){
            printf("ERROR...\n");
            break;
        }
#else
        if(RX_buf_offset > 511){
            printf("ERROR...\n");
            break;
        }
#endif

        // Populate gnr request
#ifdef SINGLE_GPU_PKT_PTR
        gnr->req_ptr = (size_t *)&RXbp[RX_buf_offset]; // Shift into RXbp buffer for request batch (Host req ptr)
#else
        gnr->req_ptr = (size_t *)&RXbp[RX_buf_offset*NUM_REQUESTS_PER_BATCH]; // Shift into RXbp buffer for request batch (Host req ptr)
#endif


#ifdef DO_GNOM_TX
        gnr->res_ptr = (size_t *)&TXbp[TX_buf_offset*NUM_REQUESTS_PER_BATCH]; // Shift into TXbp buffer for request batch (Host res ptr)
#endif

        gnr->timestamp = current_time;
        gnr->req_id = global_req_id++;
        gnr->batch_id = gpu_nom_ret[BATCH_ID];
        gnr->queue_ind = gpu_nom_ret[QUEUE_IND];

        //csm->num_batches_processed++;
        num_batches++;
#ifndef BG_TEST_ONLY
        if(num_batches == 60000){
            background_start_flag = 1;
            pthread_mutex_unlock(&csm->bg_thread_mutex); // If BG task running concurrently, lock it until ready to launch work
        }
#endif


        // Populate arguments
#ifdef SINGLE_GPU_PKT_PTR
        size_t gpu_first_pkt_ptr = *gnr->req_ptr;
        cuda_args[0] = (void *)&gpu_first_pkt_ptr;
#else
        cuda_args[0] = (void *)&gnr->gpu_req_ptr;
#endif

        cuda_args[1] = (void *)&gnr->num_req;
        cuda_args[2] = (void *)&gnr->gpu_res_ptr;
#ifdef NETWORK_ONLY_TEST
        cuda_args[3] = (void *)&gnr->timestamp;
#else
        cuda_args[6] = (void *)&gnr->timestamp;
#endif

        // Set stream
        gnr->queue_id = current_stream_index;

#ifndef USE_DIRECT_ACCESS_MEM_REQUEST

#ifndef SINGLE_GPU_PKT_PTR
        // If not SINGLE_GPU_PKT_PTR, then need to copy the full buffer of request pointers to GPU
        // Otherwise, just use first buffer pointer as a parameter to the kernel
        checkCudaErrors(cuMemcpyHtoDAsync_v2(gnr->gpu_req_ptr,
                                             gnr->req_ptr,
                                             NUM_REQUESTS_PER_BATCH*sizeof(void *),
                                             csm->m_streams[current_stream_index]));
#endif

#ifdef DO_GNOM_TX
        checkCudaErrors(cuMemcpyHtoDAsync_v2(gnr->gpu_res_ptr,
                                             gnr->res_ptr,
                                             NUM_REQUESTS_PER_BATCH*sizeof(void *),
                                             csm->m_streams[current_stream_index]));
#endif

#endif

        // Launch kernel
        // Data has already been copied to the GPU directly via GPUdirect. Launch kernel.
#ifdef NETWORK_ONLY_TEST
        checkCudaErrors(cuLaunchKernel(m_wta->cuda_ctx->network_function, blocksPerGrid, 1, 1,
                                       threadsPerBlock, 1, 1, 0,
                                       csm->m_streams[current_stream_index], cuda_args, NULL));
#else
        checkCudaErrors(cuLaunchKernel(m_wta->cuda_ctx->hcuFunction, blocksPerGrid, 1, 1,
                                       threadsPerBlock, 1, 1, 0,
                                       csm->m_streams[current_stream_index], cuda_args, NULL));
#endif

        // Copy Responses
#ifndef USE_DIRECT_ACCESS_MEM_RESPONSE
        // Copy response buffer back from GPU
        checkCudaErrors(cuMemcpyDtoHAsync_v2(gnr->res_ptr, gnr->gpu_res_ptr, BUFFER_SIZE*sizeof(int), csm->m_streams[current_stream_index]));
#endif

        // Create an event
        //checkCudaErrors(cuEventCreate(&csm->m_gpu_event_queue[current_event_index], CU_EVENT_DISABLE_TIMING));
        // Record event to signal the completion of this kernel
        checkCudaErrors(cuEventRecord(csm->m_gpu_event_queue[current_event_index], csm->m_streams[current_stream_index]));

        // Push event into map
        pthread_mutex_lock(&csm->m_map_mutex[current_stream_index]);
        m_gsp.gpu_event = &csm->m_gpu_event_queue[current_event_index];
        m_gsp.gnr = gnr;
        csm->m_k_event_queue[current_stream_index]->push(m_gsp);
        pthread_mutex_unlock(&csm->m_map_mutex[current_stream_index]);

        inc_wrap_index(current_stream_index, start_stream_index, end_stream_index);
        inc_wrap_index(current_event_index, start_event_index, end_event_index);

        average_dispatch_time = (dispatch_sum / (double)dispatch_count)*1000000.0;
    }while(!stop_stream_manager);
    csm->num_batches_processed = num_batches;
    stop_stream_manager = true;
    free(gpu_nom_ret);

    printf("Pusher thread complete... cleaning up\n");
    printf("Dispatch sum: %lf, dispatch_count: %llu\n", dispatch_sum, dispatch_count);
    average_dispatch_time = (dispatch_sum / (double)dispatch_count)*1000000.0;

    sleep(1);
    if(m_tid == 0) // Only the first thread should clean things up
        csm->stop_gpu_nom();
    sleep(2);

    return NULL;

}


void *cuda_stream_manager::gnom_post(void *arg){

    worker_thread_args *m_wta = (worker_thread_args *)arg;
    cuda_stream_manager *csm = (cuda_stream_manager *)m_wta->csm;
    int m_tid = m_wta->tid;
    size_t *RXbp = m_wta->mmaped_RX_buffer_pointers;
    size_t *TXbp = m_wta->mmaped_TX_buffer_pointers;

    checkCudaErrors(cuCtxPushCurrent(m_wta->cuda_ctx->hcuContext));

    gnom_buf_recycle_info recycle_info;

    CUresult status;
    CUevent *event;

    int m_stream_ind = 0;
    int m_stream_ind_stride = NUM_GNOM_POST_THREADS;

    /*
     int last_from_gpu_q = 0;
     int index = 0;
     */

    unsigned pkt_length = 0;

    int wt_batch_cnt = 0;
    int ret = 0;
    int res_cnt = 0;

#ifdef NETWORK_ONLY_TEST
    int hdr_size = 42;
#else
    // UDP Packet header (42 bytes) + 8 Byte Memcached header = 50 Bytes.
    int hdr_size = 50;
#endif

    int buffer_idx = 0;
    int flush_packet = 0;
    int sent_bytes = 0;



    size_t *item_ptr = NULL;
    u_char *pkt_hdr_ptr = NULL;
    u_char *pf_pkt_buffer_ptr = NULL;
    item *itm = NULL;

    pfring_zc_queue *pf_zq = NULL;
    pfring_zc_pkt_buff *pf_buf = NULL;
    pfring_zc_pkt_buff **m_pf_bufs = NULL;


    gpu_stream_pair m_gsp;
    gpu_nom_req *m_req;

    unsigned long long num_memc_misses = 0;
    unsigned long long num_memc_hits = 0;
    uint64_t num_pkt_processed = 0;

    unsigned start_pf_buffer_ind = 0;
    unsigned cur_pf_buffer_ind = 0;
    unsigned free_pf_buffer_ind = 0;

#ifdef BIND_CORE
    //    set_thread_affinity((m_tid+1) % 4); //
    //set_thread_affinity(m_tid + 2);
#endif

    pf_zq = csm->m_pf_zq[m_tid];
    m_pf_bufs = csm->m_pf_buffers[m_tid]; // Point to correct PF_RING buffers for this thread
    m_stream_ind = m_tid;

    pthread_mutex_lock( &csm->m_host_mutex );
    printf("Puller thread: %d | start_index: %d | stride: %d \n", m_tid, m_stream_ind, m_stream_ind_stride);
    pthread_mutex_unlock( &csm->m_host_mutex );


    // Each GNoM_post thread looks at a stride of queues
    // Example with 2 threads
    //      Thread 0: 0, 2, 4, 6, 8...
    //      Thread 1: 1, 3, 5, 7, 9...
    // - 4 stages.
    //      - (0) Check stream for completed kernel (no mutex, each thread has own set of streams in a strided fashion)
    //      - (1) Recycle GRX buffers (mutex 1)
    //      - (2) Populate PF_RING buffers (No mutex, each thread has own set of buffers)
    //      - (3) Send to PF_RING (mutex 2)
    do{

        // Kernels pushed in FCFS, check for first completed on a per-command queue basis
        if(!csm->m_k_event_queue[m_stream_ind]->empty()){ // Something in the stream queue

            // (0)
            m_gsp = csm->m_k_event_queue[m_stream_ind]->front();
            status = cuEventQuery(*m_gsp.gpu_event);

            if(status == CUDA_SUCCESS){ // Stream operations have completed for this stream

                csm->m_k_event_queue[m_stream_ind]->pop();
                m_req = m_gsp.gnr;

                // Recycle RX buffers to GPU-NoM
                recycle_info.batch_id = m_req->batch_id;
                recycle_info.queue_ind = m_req->queue_ind;

                // (1)
                pthread_mutex_lock(&csm->gnom_recycle_mutex);
                ret = write(m_wta->gpu_km_fp, &recycle_info, sizeof(recycle_info));
                pthread_mutex_unlock(&csm->gnom_recycle_mutex);

                //checkGPUNoMErrors(ret, 1);

                //printf("TID=%d: Batch %d complete from stream: %d!\n", m_tid, wt_batch_cnt, m_stream_ind);

                // Perform response
                item_ptr = (size_t *)m_req->res_ptr;
                pkt_hdr_ptr = (u_char *)((size_t)m_req->res_ptr + NUM_REQUESTS_PER_BATCH*sizeof(size_t)); // Pkt headers are passed the item pointers in the response memory

                // (2)
                start_pf_buffer_ind = free_pf_buffer_ind;
                cur_pf_buffer_ind = free_pf_buffer_ind;
                free_pf_buffer_ind = (free_pf_buffer_ind+NUM_REQUESTS_PER_BATCH) % NUM_PF_BUFS;

                for(unsigned i=0; i<NUM_REQUESTS_PER_BATCH; ++i){
                    pf_buf = m_pf_bufs[cur_pf_buffer_ind];
                    cur_pf_buffer_ind++;

                    // Set pf_ring buffer pointer
                    pf_pkt_buffer_ptr = pf_buf->data;

#ifdef BG_TEST
                    // If Background worklaod test, set the TTL to 0 when BG thread is complete to
                    // signal to the client the end of BG execution

                    if(likely(background_done_flag)){
                        // If the BG task is done, set ttl to 0
                        ip_header *iph = (ip_header *)((size_t)pkt_hdr_ptr + sizeof(ether_header));
                        iph->ttl = 0;
                    }else if(unlikely(background_start_flag)){
                        // If the BG task is running, set ttl to 1
                        ip_header *iph = (ip_header *)((size_t)pkt_hdr_ptr + sizeof(ether_header));
                        iph->ttl = 1;
                    }
#endif

#ifdef NETWORK_ONLY_TEST
#ifdef LATENCY_MEASURE
                    // If network only test, no payload, just send packet
                    memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, hdr_size+8); // hdr + 8 byte latency timestamp from client
                    pf_pkt_buffer_ptr+=(hdr_size+8);
#else
                    // If network only test, no payload, just send packet
                    memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, hdr_size);
                    pf_pkt_buffer_ptr+=hdr_size;

#endif
#else
                    //if(i != 0){ // Used for testing kernel execution time
                    if(item_ptr[i] != 0){ // If an item was found for this request
                        num_memc_hits++;
                        itm = (item *)item_ptr[i];

                        // Set buffer length
                        // Total packet size = Network header + "VALUE " + key + suffix + data (with "\r\n")
#ifdef CONSTANT_RESPONSE_SIZE
                        pf_buf->len = RESPONSE_SIZE;
#else
                        pf_buf->len = itm->gnom_total_response_length;

#endif


#ifdef LATENCY_MEASURE
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 52);
                        pf_pkt_buffer_ptr+= (52);
#else
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 48);
                        pf_pkt_buffer_ptr+= (48);
#endif

                        memcpy(pf_pkt_buffer_ptr, ITEM_key(itm), itm->nkey);
                        pf_pkt_buffer_ptr+= (itm->nkey);

                        // Copy Suffix + Value
                        memcpy(pf_pkt_buffer_ptr, ITEM_suffix(itm), itm->nsuffix + itm->nbytes);
                        pf_pkt_buffer_ptr += (itm->nsuffix + itm->nbytes);

                    }else{ // else, no item was found
                        num_memc_misses++;

                        pf_buf->len = RESPONSE_SIZE;

#ifdef LATENCY_MEASURE
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 52);
                        pf_pkt_buffer_ptr+= (52);
#else
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 48);
                        pf_pkt_buffer_ptr+= (48);
#endif

                        memcpy(pf_pkt_buffer_ptr, NF, strlen(NF));
                        pf_pkt_buffer_ptr += strlen(NF);
                    }
                    //}
                    // Write "END"
                    memcpy(pf_pkt_buffer_ptr, END, strlen(END));
#endif


#if 0
                    // Testing sending the packet immediately after populating pf_ring vs. populating all pf_ring buffers and then sending all responses.
                    num_pkt_processed++;
                    // Send packet
                    while( (sent_bytes = pfring_zc_send_pkt(pf_zq, &pf_buf, flush_packet)) < 0 ){
                        if(unlikely(stop_stream_manager))
                            break;
                    }
#endif

                    pkt_hdr_ptr += RESPONSE_HDR_STRIDE; // Currently 256 Bytes per response stride - Move to next response packet
                }

                // (3)
                cur_pf_buffer_ind = start_pf_buffer_ind; // Move back to first PF_RING buffer in this batch
                for(unsigned i=0; i<NUM_REQUESTS_PER_BATCH; ++i){
                    pf_buf = m_pf_bufs[cur_pf_buffer_ind];
                    cur_pf_buffer_ind++;

                    //if(likely(csm->verify_pkt_hdr((void *)pf_buf->data))){ // For testing single queue
                    // Send response packets through PF_RING
                    while( (sent_bytes = pfring_zc_send_pkt(pf_zq, &pf_buf, flush_packet)) < 0 ){
                        if(unlikely(stop_stream_manager))
                            break;
                    }
                }
                num_pkt_processed += NUM_REQUESTS_PER_BATCH;

                pfring_zc_sync_queue(pf_zq, tx_only); // Sync after a batch is sent

                wt_batch_cnt++;

                // GNoM: Ensure ordering of GET/SET requests on GPU is preserved on the CPU
                csm->m_get_complete_timestamp[m_tid] = m_req->timestamp;

                m_req->in_use = 0;

            }else{
                //usleep(1); // Avoid constant driver calls
            }
        }

        // Check the next stream allocated to this puller thread
        m_stream_ind = (m_stream_ind + m_stream_ind_stride) % (NUM_C_KERNELS);

    }while(!stop_stream_manager);

    pthread_mutex_lock( &csm->m_host_mutex );
    // Get the lock, update global stats
    total_num_pkt_processed += num_pkt_processed;
    csm->num_memc_misses += num_memc_misses;
    csm->num_memc_hits += num_memc_hits;
    stop_stream_manager = true;
    printf("Puller thread %d complete...\n", m_tid);
    pthread_mutex_unlock( &csm->m_host_mutex );

    return NULL;

}



// GPU SET request handler. Blocking function to update GPU Memcached hashtable
int cuda_stream_manager_base::set_lock(){
    pthread_mutex_lock(&m_set_mutex);
    return 0;
}

int cuda_stream_manager_base::set_unlock(){
    pthread_mutex_unlock(&m_set_mutex);
    return 0;
}


int cuda_stream_manager_base::poll_get_complete_timestamp(int set_evict_timestamp){

    bool complete = false;
    bool stall = false;

    // Wait until all of the GET request batches run concurrnetly with this SET request have completed their operations
    while(!complete){
        complete = true;
        for(unsigned i=0; i<NUM_GNOM_POST_THREADS; ++i){
            if(set_evict_timestamp > m_get_complete_timestamp[i]){
                complete = false;
                if(!stall){
                    stall = true;
                    num_conflict_set_get++;
                }
                usleep(500);
                break;
            }
        }
    }
    //printf("GNoM: SET ready to proceed with modification\n");

    return 1;

}

void cuda_stream_manager_base::inc_set_hit(){
    num_set_hit++;
}
void cuda_stream_manager_base::inc_set_miss(){
    num_set_miss++;
}
void cuda_stream_manager_base::inc_set_evict(){
    num_set_evict++;
}


int cuda_stream_manager_base::set_update_gpu_hashtable(void *req, int req_size, void *res, int res_size, rel_time_t timestamp){

    // Mutex set in calling thread
    if(!stop_stream_manager){

        if(cuCtxPushCurrent(m_context->hcuContext) != CUDA_SUCCESS)
            printf("**Error** Unable to push CTX in SET thread...\n");

        // Copy over requests
        if(cuMemcpyHtoDAsync_v2(d_set_req, req, req_size, m_set_stream) != CUDA_SUCCESS)
            printf("**Error** Unable to copy SET request batch buffer...\n");

        void *cuda_args[] = { &d_set_req, &d_set_res, (void *)&hashpower, (void *)&m_gpu_hashtable_ptr, (void *)&m_gpu_lock_ptr, (void *)&timestamp };

        // Launch SET kernel
        if(cuLaunchKernel(m_context->set_function, 1, 1, 1, 32, 1, 1, 0, m_set_stream, (void **)cuda_args, NULL) != CUDA_SUCCESS)
            printf("**Error** worker thread unable to launch CUDA SET kernel...\n");

        // Pull back the response
        if(cuMemcpyDtoHAsync_v2(res, d_set_res, res_size, m_set_stream) != CUDA_SUCCESS)
            printf("**Error** worker thread unable to copy response batch buffer...\n");

        // Wait for completion
        if(cuStreamSynchronize(m_set_stream) != CUDA_SUCCESS)
            printf("**Error** worker thread unable sync SET stream...\n");

    }

    cuCtxPopCurrent(NULL);
    return 0;
}

// Helper function to verify that the UDP port is correct
int cuda_stream_manager_base::verify_pkt_hdr(void *hdr){
    ether_header *eh = (ether_header *)hdr;
    ip_header *iph = (ip_header *)((size_t)hdr + sizeof(ether_header));
    udp_header *uh = (udp_header *)((size_t)hdr + sizeof(ether_header) + sizeof(ip_header));

    // Debugging
    if(ntohs(uh->dest) != 50000){
        //printf("WRONG UDP PORT: dst = %hu, source = %hu\n", ntohs(uh->dest), ntohs(uh->source));
        return 0;
    }else{
        return 1;
    }

}

// Helper function to print out packet header data
void cuda_stream_manager_base::print_pkt_hdr(void *hdr){
    ether_header *eh = (ether_header *)hdr;
    ip_header *iph = (ip_header *)((size_t)hdr + sizeof(ether_header));
    udp_header *uh = (udp_header *)((size_t)hdr + sizeof(ether_header) + sizeof(ip_header));
    uint8_t *payload = (uint8_t *)((size_t)hdr + sizeof(ether_header) + sizeof(ip_header) + sizeof(udp_header));

    printf("Packet header contents: \n");

    /***** ETHERNET HEADER *****/
    printf("\t==Ethernet header==\n");
    printf("\t\tDest: ");
    for(unsigned i=0; i<ETH_ALEN; ++i)
        printf("%hhx ", eh->ether_dhost[i]);
    printf("\n\t\tSource: ");
    for(unsigned i=0; i<ETH_ALEN; ++i)
        printf("%hhx ", eh->ether_shost[i]);
    printf("\n\t\tType: %hx\n", eh->ether_type);
    /***** END ETHERNET HEADER *****/

    /***** IP HEADER *****/
    printf("\t==IP header==\n");
    printf("\t\tVersion+hdr_len: %hhu\n", iph->version);
    printf("\t\tTOS: %hhu\n", iph->tos);
    printf("\t\tTotal Length: %hu\n", ntohs(iph->tot_len));
    printf("\t\tID: %hu\n", ntohs(iph->id));
    printf("\t\tFrag_off: %hu\n", iph->frag_off);
    printf("\t\tTTL: %hhu\n", iph->ttl);
    printf("\t\tProtocol: %hhu\n", iph->protocol);
    printf("\t\tchecksum: %hu\n", ntohs(iph->check));
    printf("\t\tSource address: %x\n", ntohl(iph->saddr));
    printf("\t\tDest address: %x\n", ntohl(iph->daddr));
    /***** END IP HEADER *****/

    /***** UDP HEADER *****/
    printf("\t==UDP header==\n");
    printf("\t\tSource port: %hu\n", ntohs(uh->source));
    printf("\t\tDest port: %hu\n", ntohs(uh->dest));
    printf("\t\tLength: %hu\n", ntohs(uh->len));
    printf("\t\tChecksum: %hu\n", uh->check);
    /***** END UDP HEADER *****/

    printf("\nPayload: ");
    for(unsigned i=0; i<(PKT_STRIDE-42); ++i){
        printf("%02x", (payload[i]));
    }


}

bool cuda_stream_manager_base::signal_gpu_km(){
    int res = 0;

    cout << "Signaling GNoM-ND that GNoM-host is successfully configured...";
    res = ioctl(m_gpu_km_fp, SIGNAL_NIC, NULL);
    if(res != 0){
        cout <<"ioctl_err: abort()" << endl;
        cuCtxPopCurrent(NULL);
        return false;
    }
    cout << "Complete: " << res << endl;
    return true;
}

bool cuda_stream_manager_base::reg_gpu_km_grxb(kernel_args_hdr *ka){
    int res = 0;

    // Register GPU CUDA RX buffers
    res = ioctl(m_gpu_km_fp, GPU_REG_MULT_BUFFER_CMD, ka);
    if(res != 0){
        cout <<"ioctl_err: abort()" << endl;
        cuCtxPopCurrent(NULL);
        return false;
    }

    cout << "Registering GRXBs complete..." << endl;
    return true;
}

bool cuda_stream_manager_base::reg_gpu_km_gtxb(kernel_args_hdr *ka){
    int res=0;

    // Pass GPU-accessible CPU TX buffers to GNoM_km
    res = ioctl(m_gpu_km_fp, GNOM_REG_MULT_CPU_BUFFERS, ka);
    if(res != 0){
        cout <<"ioctl_err: abort() on GNOM_REG_MULT_CPU_BUFFERS" << endl;
        cuCtxPopCurrent(NULL);
        return false;
    }

    cout << "Registering GTXBs complete..." << endl;
    return true;
}

/****** GPUDirect ******/
bool cuda_stream_manager_base::init_gpu_km(){
    int res=0;
    CUresult status;

    cout << "Initializing GPU_KM... ";

    // Open GPU_km. Pin, map, and register CUDA buffers with GPU_km
    m_dev = "/dev/gpu_km";
    m_gpu_km_fp = 0;

    m_gpu_km_fp = open (m_dev.c_str(), O_RDWR);

    if (m_gpu_km_fp == -1){
        cout << "Unable to open " << m_dev << endl;
        cuCtxPopCurrent(NULL);
        return false;
    }

    sleep(1);

    cout << "Complete" << endl;


    fflush(stdout);

    return true;
}




void cuda_stream_manager_base::print_stats(){

    printf("\n\n======= SETs =======\n");
    printf("Total number of SETs: %llu\n", num_set_hit + num_set_miss + num_set_evict);
    printf("Total number hit: %llu\n", num_set_hit);
    printf("Total number miss: %llu\n", num_set_miss);
    printf("Total number evict: %llu\n", num_set_evict);
    printf("Total number of conflicting SET stalls: %llu\n", num_conflict_set_get);
    printf("=================\n");
    printf("\n======= GETs =======\n");
    printf("total number of requests processed (received and sent) from GNoM: %lu\n", total_num_pkt_processed);
    printf("Total number of batches processed: %llu\n", num_batches_processed);
    printf("Total number of Memcached hits: %llu\n", num_memc_hits);
    printf("Total number of Memcached misses: %llu\n", num_memc_misses);
    printf("Hit rate: %lf\n\n", (double)num_memc_hits/((double)(num_memc_hits+num_memc_misses)));
    printf("=================\n");


    /*
    for(unsigned i=0; i<NUM_C_KERNELS; ++i){
        printf("Average kernel times for stream %i = %lf us\n", i, (double)total_time_per_stream[i]/(double)count_per_stream[i]);
    }
    */

    printf("Average dispatch rate = %lf us per batch\n", average_dispatch_time);
}

void cuda_stream_manager_base::stop(){
    stop_gpu_nom();
}

void cuda_stream_manager::stop_gpu_nom(){
    int res = 0;
    static int called=0;
    // Ensure stop_stream_manager is set
    stop_stream_manager = true;

    cout << "Stop called, sleeping for 3 seconds to wait for global stats to update." << endl;
    sleep(3); // Wait for stat updates

    pthread_mutex_lock(&m_host_mutex);

    if(!called){ // Ensure only 1 thread does this
        print_stats(); // Print stats

        cout << "Signaling GPU-NoM to shutdown..." << endl;
        res = ioctl(m_gpu_km_fp, STOP_SYSTEM, NULL);
        cout << "Complete..." << res << endl;

        // Unhook NIC from GPU-NoM
        cout << "Shutting down NIC...";
        res = ioctl(m_gpu_km_fp, SHUTDOWN_NIC, NULL);
        cout << "Complete..." << res << endl;

#ifdef USE_PF_RING
        for(unsigned i=0; i<MULTI_PF_RING_TX; ++i){
            cout << "Closing PF_RING " << i << "...";
            pfring_zc_destroy_cluster(m_pf_zc[i]); // Close PF_RING
            cout << "Complete..." << endl;
        }
#endif

        // Unmap request / response ptr buffers
        cout << "Unmapping request/response pointer buffers" << endl;
        res = munmap(mmaped_req_ptr, mmaped_req_res_ptr_len);
        cout << "Complete..." << res << endl;


        // Unmap all GPU-accessible CPU buffers
#ifdef DO_GNOM_TX
        res = ioctl(m_gpu_km_fp, GNOM_UNREG_MULT_CPU_BUFFERS, &m_tx_kernel_arg_hdr);
        if(res != 0){
            cout <<"ioctl_err: abort() on GNOM_REG_MULT_CPU_BUFFERS" << endl;
        }

#endif

        cout << "Closing GPU-NoM...";
        close(m_gpu_km_fp); // Close the GPU-NoM file descriptor
        cout << "Complete..." << endl;

        // Destroy all CUDA event objects
        for(unsigned i=0; i<MAX_NUM_BATCH_INFLIGHT; ++i){
            cuEventDestroy(m_gpu_event_queue[i]);
        }

        // Free GRXBs
        cout << "Freeing GRXBs...";
#ifdef SINGLE_GPU_BUFER
        if(m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr){
            checkCudaErrors(cuMemFree(m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr));
        }
#else
        for(unsigned i=0; i<NUM_GPU_PAGES; ++i){
            if(m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_addr)
                checkCudaErrors(cuMemFree(m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_addr));
        } // GPU Pages initialized
#endif

        cout << "Complete..." << endl;


#ifdef DO_GNOM_TX
        // Free GTXBs
        cout << "Freeing GTXBs...";
        for(unsigned i=0; i<NUM_GPU_TX_PAGES; ++i){
            if(m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va){
                checkCudaErrors(cuMemFreeHost(m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va));
            }
        } // GPU Pages initialized
        cout << "Complete..." << endl;

#endif
        // Free Kernel argument headers
        cout << "Freeing kernel arguement headers...";
        if(m_kernel_arg_hdr.buf_meta_data.gpu_args)
            free(m_kernel_arg_hdr.buf_meta_data.gpu_args);
        cout << "Complete..." << endl;

#ifdef DO_GNOM_TX
        if(m_tx_kernel_arg_hdr.buf_meta_data.cpu_args)
            free(m_tx_kernel_arg_hdr.buf_meta_data.cpu_args);
#endif


        // Freeing SET request buffers
         cout << "Freeing SET buffers...";
         if(d_set_req)
             checkCudaErrors(cuMemFree(d_set_req));

         cout << "req complete...";
         fflush(stdout);
         sleep(2);


         if(d_set_res)
             checkCudaErrors(cuMemFree(d_set_res));

         cout << "Res Complete..." << endl;
         fflush(stdout);
         sleep(1);

        cout << "Complete..." << endl;
    }
    called = 1;

    pthread_mutex_unlock(&m_host_mutex);

//    cuProfilerStop();
    cudaDeviceReset();

    exit(EXIT_SUCCESS);


    if(m_gpu_nom_req)
        delete m_gpu_nom_req;

    if(gpu_nom_response_buffers)
        delete gpu_nom_response_buffers;

    if(host_response_buffers)
        delete host_response_buffers;

    if(m_gpu_event_queue)
        delete m_gpu_event_queue;


}


// Background thread used to stress GPU with additional background task during Memcached execution
void *cuda_stream_manager_base::bg_thread_main(void *arg){

    struct timespec start_t, end_t;
    struct timespec start_kernel, end_kernel;
    double cur_time = 0, kernel_time = 0;
    bg_thread_args *m_args = (bg_thread_args *)arg;
    cuda_stream_manager_base *csm = (cuda_stream_manager_base *)m_args->csm;
    int m_tid = m_args->tid;

    checkCudaErrors(cuCtxPushCurrent(m_args->cuda_ctx->hcuContext));

    int threads_per_block = 1024;
    int blocks_per_grid = BLOCKS_PER_GRID;
    int multiplier = 4;
    int num_iterations = 1000; //m_args->num_iterations;

    CUdeviceptr d_A, d_B, d_C;
    int *h_A, *h_B, *h_C;

    int total_array_size = BLOCKS*threads_per_block*multiplier*sizeof(int); // # CTA * # thread/CTA * multiplier

    h_A = (int *)malloc(total_array_size);
    h_B = (int *)malloc(total_array_size);
    h_C = (int *)malloc(total_array_size);

    // Allocate GPU resources
    if(cuMemAlloc(&d_A, total_array_size) != CUDA_SUCCESS)
        printf("**ERROR** Couldn't allocate d_A...\n");

    if(cuMemAlloc(&d_B, total_array_size) != CUDA_SUCCESS)
        printf("**ERROR** Couldn't allocate d_B...\n");

    if(cuMemAlloc(&d_C, total_array_size) != CUDA_SUCCESS)
        printf("**ERROR** Couldn't allocate d_C...\n");


    // Init host buffers
    int array_length = total_array_size / sizeof(int);
    for(unsigned i = 0; i < array_length; ++i){
        h_A[i] = i*i + i*i;
        h_B[i] = i*i - i;
        h_C[i] = 0;
    }

    int dummy0 = 0;
    void *cuda_args[] = {(void *)&d_A, (void *)&d_B, (void *)&d_C, (void *)&array_length, (void *)&num_iterations, (void *)&dummy0};

    printf("BG thread ready to launch kernel, obtaining lock...\n");
    pthread_mutex_lock(&csm->bg_thread_mutex);


#ifndef BG_TEST_ONLY
    // Start timer
    clock_gettime(CLOCK_REALTIME, &start_t);

    // Init GPU buffers
    checkCudaErrors(cuMemcpyHtoDAsync(d_A, h_A, total_array_size, csm->m_set_stream));
    checkCudaErrors(cuMemcpyHtoDAsync(d_B, h_B, total_array_size, csm->m_set_stream));

    // Start timer
    clock_gettime(CLOCK_REALTIME, &start_kernel);


    for(unsigned i=0; i<DIVISOR; ++i){

        // Launch kernel
        checkCudaErrors(cuLaunchKernel(m_args->cuda_ctx->background_function, blocks_per_grid, 1, 1,
                                       threads_per_block, 1, 1, 0, csm->m_set_stream, cuda_args, NULL));

        // Wait for completion
        checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));

    }

    background_done_flag = 1;
    // End timer
    clock_gettime(CLOCK_REALTIME, &end_kernel);


    // Copy back result
    checkCudaErrors(cuMemcpyDtoHAsync(h_C, d_C, total_array_size, csm->m_set_stream));

    // Wait for completion
    checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));

    // End timer
    clock_gettime(CLOCK_REALTIME, &end_t);
    cur_time = (double)((end_t.tv_sec - start_t.tv_sec) + ((end_t.tv_nsec - start_t.tv_nsec)/1E9));
    kernel_time = (double)((end_kernel.tv_sec - start_kernel.tv_sec) + ((end_kernel.tv_nsec - start_kernel.tv_nsec)/1E9));

    /*
    cur_time = cur_time * 1000000.0;
    kernel_time = kernel_time * 1000000;
    printf("BG CUDA total runtime = %.10lf uSec, kernel runtime = %.10lf\n", cur_time, kernel_time);
    */
#else

    // warmup kernel
    for(unsigned i=0; i<10; ++i){
        checkCudaErrors(cuMemcpyHtoDAsync(d_A, h_A, total_array_size, csm->m_set_stream));
        checkCudaErrors(cuMemcpyHtoDAsync(d_B, h_B, total_array_size, csm->m_set_stream));

        for(unsigned i=0; i<DIVISOR; ++i){

            // Launch kernel
            checkCudaErrors(cuLaunchKernel(m_args->cuda_ctx->background_function, blocks_per_grid, 1, 1,
                                           threads_per_block, 1, 1, 0, csm->m_set_stream, cuda_args, NULL));

            // Wait for completion
            checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));

        }
        checkCudaErrors(cuMemcpyDtoHAsync(h_C, d_C, total_array_size, csm->m_set_stream));
        checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));
    }


    // Run test
    printf("Now running test...\n");

    blocks_per_grid = 256;
    threads_per_block = 1024;

    int total_blocks = 256;
    int divisor = 1;


    while(divisor <= 256){
        printf("blocks_per_grid: %d, divisor: %d\n", blocks_per_grid, divisor);

        checkCudaErrors(cuMemcpyHtoDAsync(d_A, h_A, total_array_size, csm->m_set_stream));
        checkCudaErrors(cuMemcpyHtoDAsync(d_B, h_B, total_array_size, csm->m_set_stream));

        // Start timer
        clock_gettime(CLOCK_REALTIME, &start_kernel);
        for(unsigned i=0; i<divisor; ++i){
            // Launch kernel
            checkCudaErrors(cuLaunchKernel(m_args->cuda_ctx->background_function, blocks_per_grid, 1, 1,
                                           threads_per_block, 1, 1, 0, csm->m_set_stream, cuda_args, NULL));

            // Wait for completion
            checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));
        }
        checkCudaErrors(cuMemcpyDtoHAsync(h_C, d_C, total_array_size, csm->m_set_stream));
        checkCudaErrors(cuStreamSynchronize(csm->m_set_stream));

        // End timer
        clock_gettime(CLOCK_REALTIME, &end_kernel);
        kernel_time = (double)((end_kernel.tv_sec - start_kernel.tv_sec) + ((end_kernel.tv_nsec - start_kernel.tv_nsec)/1E9));

        kernel_time = kernel_time * 1000000;
        printf("BG kernel runtime (%d) = %.10lf\n", 256/divisor, kernel_time);


        divisor = divisor << 1;
        blocks_per_grid = total_blocks/divisor;
    }

#endif

    sleep(15);

    cur_time = cur_time * 1000000.0;
    kernel_time = kernel_time * 1000000;
    printf("BG CUDA total runtime = %.10lf uSec, kernel runtime = %.10lf\n", cur_time, kernel_time);
    pthread_mutex_unlock(&csm->bg_thread_mutex);

}


////////////////////////////////////////////////////////
////////////////////////// CSM /////////////////////////
////////////////////////////////////////////////////////

/// Destructor
cuda_stream_manager::~cuda_stream_manager(){
    // Cleanup queues
    stop_gpu_nom();
}

/// Constructor
cuda_stream_manager::cuda_stream_manager(CUDAContext *context,
                                         CUdeviceptr gpu_hashtable_ptr,
                                         CUdeviceptr gpu_lock_ptr) :
cuda_stream_manager_base(context, gpu_hashtable_ptr, gpu_lock_ptr){

    if(global_csm == NULL){
        global_csm = this;
    }else{
        cout << "Error: Global_csm is already set... multiple invocations of cuda_stream_manager?" << endl;
        abort();
    }
    
    // Push context for this thread
    CUresult status = cuCtxPushCurrent(m_context->hcuContext);
    if(status != CUDA_SUCCESS){
        printf("Stream_manager_init: Context push error: %d\n", status);
        exit(1);
    }

    /*************** Subclass specific ***************/


    m_gpu_nom_req = new gpu_nom_req[MAX_NUM_BATCH_INFLIGHT];
    gpu_nom_response_buffers = new CUdeviceptr[MAX_NUM_BATCH_INFLIGHT];
    host_response_buffers = new size_t*[MAX_NUM_BATCH_INFLIGHT];
    m_gpu_event_queue = new CUevent[MAX_NUM_BATCH_INFLIGHT];

    // Create all the CUDA event objects
    for(unsigned i=0; i<MAX_NUM_BATCH_INFLIGHT; ++i){
         checkCudaErrors(cuEventCreate(&m_gpu_event_queue[i], CU_EVENT_DISABLE_TIMING));
    }

    // Initialize GNoM
    if(init_gpu_km() != true){
        printf("Error: init_gpu_km failed...\n");
    }

    int req_buf_size = NUM_REQUESTS_PER_BATCH*sizeof(void *);
    int res_buf_size = (RESPONSE_HDR_STRIDE+sizeof(void *))*NUM_REQUESTS_PER_BATCH;

    
    // GRX buffers
#ifdef SINGLE_GPU_BUFER
    // 220 MB of RX buffers
    m_kernel_arg_hdr.num_pages = NUM_GPU_PAGES;
    m_kernel_arg_hdr.num_buffers = NUM_GPU_BUFFERS;

    m_kernel_arg_hdr.buffer_type = 0;
    m_kernel_arg_hdr.buf_meta_data.gpu_args = (kernel_args *)malloc(sizeof(kernel_args)); // Allocate total # of pages

    m_kernel_arg_hdr.buf_meta_data.gpu_args->m_size = NUM_GPU_BUFFERS*RX_BUFFER_SZ;//1024*1024*220;

    if(cuMemAlloc_v2((long long unsigned int *)&m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr, m_kernel_arg_hdr.buf_meta_data.gpu_args->m_size) != CUDA_SUCCESS){
        printf("Error: Failed allocating RX page %d...\n", 0);
        cuCtxPopCurrent(NULL);
        abort();
    }

    if((size_t)m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr % (64*1024) != 0){
        printf("CUDA address is not 64KB aligned.\n");
    }

    status = cuPointerGetAttribute(&m_kernel_arg_hdr.buf_meta_data.gpu_args->m_tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr);
    if(status != CUDA_SUCCESS){
        printf("Error: Failed cuPointerGetAttribute (%d) %d...\n", 0, status);
        // TODO: Clean up buffers
        cuCtxPopCurrent(NULL);
        abort();
    }

    printf("Allocating %llu MB of data for buffer %p...\n", (unsigned long long)(m_kernel_arg_hdr.buf_meta_data.gpu_args->m_size / (1024*1024)), (void *)m_kernel_arg_hdr.buf_meta_data.gpu_args->m_addr);

#else
    m_kernel_arg_hdr.num_pages = NUM_GPU_PAGES;
    m_kernel_arg_hdr.num_buffers = NUM_GPU_BUFFERS;
    m_kernel_arg_hdr.buf_meta_data.gpu_args = (kernel_args *)malloc(NUM_GPU_PAGES * sizeof(kernel_args)); // Allocate total # of pages
    m_kernel_arg_hdr.buffer_type = 0;
    for(unsigned i=0; i<NUM_GPU_PAGES; ++i){
        m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_size = RX_PAGE_SZ;

        if(cuMemAlloc_v2((long long unsigned int *)&m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_addr, RX_PAGE_SZ) != CUDA_SUCCESS){
            printf("Error: Failed allocating RX page %d...\n", i);
            cuCtxPopCurrent(NULL);
            return false;
        }

        if((size_t)m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_addr % (64*1024) != 0){
            printf("CUDA ADDRESS IS NOT 64KB aligned..........(%d) \n", i);
        }

        status = cuPointerGetAttribute(&m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS, m_kernel_arg_hdr.buf_meta_data.gpu_args[i].m_addr);
        if(status != CUDA_SUCCESS){
            printf("Error: Failed cuPointerGetAttribute (%d) %d...\n", i, status);
            // TODO: Clean up buffers
            cuCtxPopCurrent(NULL);
            return false;
        }
    } // GPU Pages initialized
#endif

    // Register GRXBs with GNoM
    if(!reg_gpu_km_grxb(&m_kernel_arg_hdr)){
        printf("Error: reg_gpu_km_grxb failed...\n");
        abort();
    }


#ifdef USE_PF_RING
    int cluster_id=0;
    gnom_init_pf_ring_tx(cluster_id);

#else // ELSE NOT USE_PF_RING
#ifdef DO_GNOM_TX
    // GNoM TX - GTXB
    m_tx_kernel_arg_hdr.num_pages = NUM_GPU_TX_PAGES;
    m_tx_kernel_arg_hdr.num_buffers = NUM_GPU_TX_BUFFERS;
    m_tx_kernel_arg_hdr.buf_meta_data.cpu_args = (cpu_kernel_args *)malloc(NUM_GPU_TX_PAGES * sizeof(cpu_kernel_args)); // Allocate total # of pages

    m_tx_kernel_arg_hdr.buffer_type = 1; // TX buffers
    for(unsigned i=0; i<NUM_GPU_TX_PAGES; ++i){
        m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].page_size = TX_PAGE_SZ;
        checkCudaErrors(cuMemAllocHost((void **)&m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va, TX_PAGE_SZ)); // Host
        checkCudaErrors(cuMemHostGetDevicePointer_v2(&m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].cuda_page_va, m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va, 0)); // Device

        res_cuda_to_host_map[m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].cuda_page_va] = m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va;

        if(i < 5){
            cout << i << ": Cuda buffer: " <<
            (void *)m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].cuda_page_va <<
            "  Host buffer: " << m_tx_kernel_arg_hdr.buf_meta_data.cpu_args[i].user_page_va << endl;
        }
    }

    // Pass GPU-accessible CPU TX buffers to GNoM_km
    reg_gpu_km_gtxb(&m_tx_kernel_arg_hdr);

#else // ELSE NOT DO_GNOM_TX

    cout << "Error: Must set USE_PF_RING or DO_GNOM_TX" << endl;
    abort();

#endif // END DO_GNOM_TX
#endif // END USE_PF_RING

    /***********************************************************************************************************/

#ifdef SINGLE_GPU_PKT_PTR
    mmaped_req_res_ptr_len=MAX_NUM_BATCH_INFLIGHT*sizeof(void *); // (512 Req/batch) * (8 Bytes / pointer) for Rx = 4KB
#else
    mmaped_req_res_ptr_len=NUM_REQUESTS_PER_BATCH*MAX_NUM_BATCH_INFLIGHT*sizeof(void *); // (512 Req/batch) * (512 Entries in queue) * (8 Bytes / pointer) for Rx = 2MB
#endif

#ifdef DO_GNOM_TX
    mmaped_req_res_ptr_len *= 2; //  x2 for Rx and Tx
#endif

    mmaped_req_ptr = mmap(NULL, mmaped_req_res_ptr_len, PROT_READ, MAP_SHARED, m_gpu_km_fp, 0);

#ifdef DO_GNOM_TX
    mmaped_res_ptr = (void *)((size_t)mmaped_req_ptr + (mmaped_req_res_ptr_len/2));
#endif

    cout << "Mapping RX/TX buffer pointers complete... <" <<  mmaped_req_ptr << ">" << endl;

    /***********************************************************************************************************/


#ifdef BG_TEST
    bg_args.tid = 0;
    bg_args.cuda_ctx = m_context;
    bg_args.csm = (void *)this;
    bg_args.num_iterations = 1;

    if(pthread_create(&bg_thread, NULL, &cuda_stream_manager_base::bg_thread_main, (void *)&bg_args)){
        printf("**ERROR**: Cannot create background test thread... \n");
        abort();
    }
    
#endif


// If only going to run the background thread, don't initialize anything else
#ifndef BG_TEST_ONLY

    pthread_mutex_lock(&bg_thread_mutex); // If BG task running concurrently, lock it until ready to launch work


    if(!gnom_init_req_res_buffers(MAX_NUM_BATCH_INFLIGHT,
                                  req_buf_size,
                                  res_buf_size)){
        cout << "Error: Failed to initialize Request and Response GNoM buffers..." << endl;
        abort();
    }

    // Launch GNoM-pre and GNoM-post host threads
    gnom_launch_pre_post_threads(&gnom_pre, &gnom_post);
    
    
#else // Else BG_TEST_ONLY DEFINED
    pthread_join(bg_thread, NULL); // Wait for BG test to complete
#endif // End BG_TEST_ONLY
    
    if(!signal_gpu_km()){
        stop_gpu_nom();
        abort();
    }

    fflush(stdout);

    cuCtxPopCurrent(NULL);

}

////////////////////////////////////////////////////////
////////////////////////// NGD /////////////////////////
////////////////////////////////////////////////////////

/// Destructor
cuda_stream_manager_ngd::~cuda_stream_manager_ngd(){
    stop_gpu_nom();
}

/// Constructor
cuda_stream_manager_ngd::cuda_stream_manager_ngd(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr) :
        cuda_stream_manager_base(context, gpu_hashtable_ptr, gpu_lock_ptr) {
    
    int cluster_id = 0;
    
    int req_buf_size = NUM_REQUESTS_PER_BATCH*sizeof(void *);
    int res_buf_size = (RESPONSE_HDR_STRIDE+sizeof(void *))*NUM_REQUESTS_PER_BATCH;

    if(global_csm == NULL){
        global_csm = this;
    }else{
        cout << "Error: Global_csm is already set... multiple invocations of cuda_stream_manager?" << endl;
        abort();
    }

    // Push context for this thread
    CUresult status = cuCtxPushCurrent(m_context->hcuContext);
    if(status != CUDA_SUCCESS){
        printf("Stream_manager_init: Context push error: %d\n", status);
        exit(1);
    }

    pthread_mutex_init(&m_ngd_free_mutex, NULL);
    pthread_mutex_init(&m_pfring_rx_mutex, NULL);

    m_gpu_nom_req = new gpu_nom_req[MAX_NUM_BATCH_INFLIGHT];
    gpu_nom_response_buffers = new CUdeviceptr[MAX_NUM_BATCH_INFLIGHT];
    host_response_buffers = new size_t*[MAX_NUM_BATCH_INFLIGHT];
    m_gpu_event_queue = new CUevent[MAX_NUM_BATCH_INFLIGHT];


    // Create all the CUDA event objects
    for(unsigned i=0; i<MAX_NUM_BATCH_INFLIGHT; ++i){
         checkCudaErrors(cuEventCreate(&m_gpu_event_queue[i], CU_EVENT_DISABLE_TIMING));
    }

    // Initialize PF_RING for TX and RX
    gnom_init_pf_ring_tx(cluster_id);
    gnom_init_pf_ring_rx(cluster_id + MULTI_PF_RING_TX);

    if( !gnom_init_grxb() ){
        cout << "Error: Failed to initialize GNoM RX buffers..." << endl;
        abort();
    }
    

    if(!gnom_init_req_res_buffers(MAX_NUM_BATCH_INFLIGHT,
                                  req_buf_size,
                                  res_buf_size)){
        cout << "Error: Failed to initialize Request and Response GNoM buffers..." << endl;
        abort(); 
    }
   
    // Populate ngd_req list with response buffers 
    assert(m_n_batches == MAX_NUM_BATCH_INFLIGHT);

    deque< ngd_req* >::iterator it;
    unsigned index = 0;
    for(it = m_ngd_free_reqs.begin(); it != m_ngd_free_reqs.end(); ++it, ++index){
        (*it)->h_res_ptr = host_response_buffers[index];
        (*it)->d_res_ptr = gpu_nom_response_buffers[index];
        (*it)->m_gnom_req = &m_gpu_nom_req[index];
    }
    assert(index == MAX_NUM_BATCH_INFLIGHT);
    gnom_launch_pre_post_threads(&gnom_pre, &gnom_post);

}

bool cuda_stream_manager_ngd::gnom_init_grxb(){
    CUresult cu_res = CUDA_SUCCESS;

    // Allocate GPU buffer space for offline GRX buffers.
    // Using same structures as GNoM, but a bit differently.
    m_kernel_arg_hdr.num_pages = 0;
    m_kernel_arg_hdr.num_buffers = MAX_NUM_BATCH_INFLIGHT * NUM_REQUESTS_PER_BATCH;

    m_kernel_arg_hdr.buffer_type = 0;
    m_kernel_arg_hdr.buf_meta_data.cpu_args = (cpu_kernel_args *)malloc(sizeof(cpu_kernel_args)); // Allocate total # of pages

#ifndef USE_DIRECT_ACCESS_MEM_REQUEST

    unsigned total_size = m_kernel_arg_hdr.num_buffers * RX_BUFFER_SZ;

    printf("Attempting to allocate %llu MB of data for GRXBs\n",
            (unsigned long long)(total_size / (1024*1024)));

    m_kernel_arg_hdr.buf_meta_data.cpu_args->user_page_va = (void *)malloc(total_size);

    if( (cu_res = cuMemAlloc_v2((long long unsigned int *)&m_kernel_arg_hdr.buf_meta_data.cpu_args->cuda_page_va,
                    total_size)) != CUDA_SUCCESS){
        printf("Error: Failed allocating RX page %d...\n", cu_res);
        cuCtxPopCurrent(NULL);
        abort();
    }
    if((size_t)m_kernel_arg_hdr.buf_meta_data.cpu_args->cuda_page_va % (64*1024) != 0)
        printf("CUDA ADDRESS IS NOT 64KB aligned..........(%d) \n", 0);


    // Initialize the NGD GRXB free list
    m_n_batches = total_size / (NUM_REQUESTS_PER_BATCH * RX_BUFFER_SZ);

    printf("Allocated %llu MB of data for buffer %p with %d batches\n",
            (unsigned long long)(total_size / (1024*1024)), 
            (void *)m_kernel_arg_hdr.buf_meta_data.cpu_args->user_page_va, m_n_batches);

    
    ngd_req *t_req;
    int cur_offset = 0;
    int count = 0;
    for( unsigned i=0; i<m_n_batches; ++i ){
        t_req = new ngd_req;
        t_req->h_ptr = (void *)((size_t)m_kernel_arg_hdr.buf_meta_data.cpu_args->user_page_va + cur_offset);
        t_req->d_ptr = (CUdeviceptr)((size_t)m_kernel_arg_hdr.buf_meta_data.cpu_args->cuda_page_va + cur_offset);
        t_req->n_req = 0;
        t_req->is_free = true;

        t_req->h_res_ptr = NULL;
        t_req->d_res_ptr = (CUdeviceptr)NULL;

        //m_ngd_free_reqs.push(t_req); 
        m_ngd_free_reqs.push_back(t_req);

        cur_offset += (NUM_REQUESTS_PER_BATCH * RX_BUFFER_SZ);
        count++;
        assert(cur_offset < total_size);
    }

#else
    printf("\nError: NGD not configured for zero-copy memory memory...\n");
    abort();
#endif

    printf("Initialized %d batches in the NGD free list\n", count);
    return true;
}


bool cuda_stream_manager_ngd::gnom_ngd_read(ngd_req **rb, int tid){
    assert(tid < NUM_GNOM_PRE_THREADS);

    int pf_buf_ind = m_pf_buf_ind[tid];

    ngd_req *t_rb = NULL;
    pthread_mutex_lock(&m_ngd_free_mutex);

    // Grab an NGD buffer from the free list
    while( m_ngd_free_reqs.empty() ){
        pthread_mutex_unlock(&m_ngd_free_mutex);
        usleep(2);
        pthread_mutex_lock(&m_ngd_free_mutex);
    }

    t_rb = m_ngd_free_reqs.front();
    m_ngd_free_reqs.pop_front();
    t_rb->is_free = false;
    pthread_mutex_unlock(&m_ngd_free_mutex);

    void *req_ptr = t_rb->h_ptr; 
    int pkt_count = 0;
    int tot_size = 0;
    unsigned cur_index = 0;

    // PF_RING Receive a batch
    // TODO: Add a timer here to exit early if batch creation is taking too long
    // TODO: Currently only works with a single Memcached request size (to test out NGD framework).
    //       Need to add a header to the buffer to know index of each packet in the request buffer.
    while(likely(pkt_count < NUM_REQUESTS_PER_BATCH)){
        pfring_zc_pkt_buff **batch = &m_pf_buffers_rx[tid][pf_buf_ind];
        int recv = 0;
        recv = pfring_zc_recv_pkt_burst (m_pf_zq_rx[0], batch, (NUM_REQUESTS_PER_BATCH - pkt_count), true);
        
        pkt_count += recv;
        //printf("Received %d. Total =  %d\n", recv, pkt_count);
        unsigned i; 
        for(i=cur_index; i<(cur_index + recv); ++i){
            if(likely(m_pf_buffers_rx[tid][pf_buf_ind]->len == PKT_STRIDE)){
                memcpy(req_ptr, m_pf_buffers_rx[tid][pf_buf_ind]->data, m_pf_buffers_rx[tid][pf_buf_ind]->len); // Copy over packet data
                tot_size += m_pf_buffers_rx[tid][pf_buf_ind]->len;
                req_ptr = (void *)((size_t)req_ptr + m_pf_buffers_rx[tid][pf_buf_ind]->len); // Move to next location to store packet

            }
            pf_buf_ind = (pf_buf_ind + 1) % NUM_PF_BUFS; // Move to next PF_RING buffer
        }

    }

    m_pf_buf_ind[tid] = pf_buf_ind; // Update static structure

    t_rb->n_req = pkt_count;
    t_rb->tot_size = tot_size;

    *rb = t_rb;
    
    return true;
}

bool cuda_stream_manager_ngd::gnom_ngd_write(ngd_req *rb){
    pthread_mutex_lock(&m_ngd_free_mutex);

    rb->is_free = true;
    m_ngd_free_reqs.push_back(rb);

    pthread_mutex_unlock(&m_ngd_free_mutex);
    return true;
}


void *cuda_stream_manager_ngd::gnom_pre(void *arg){
    worker_thread_args *m_wta = (worker_thread_args *)arg;
    cuda_stream_manager_ngd *csm = (cuda_stream_manager_ngd *)m_wta->csm;
    int m_tid = m_wta->tid;
    int res_buf_size = (RESPONSE_HDR_STRIDE+sizeof(void *))*NUM_REQUESTS_PER_BATCH;


    int m_stream_index = m_tid; // All start at an offset, increment by m_stride. 
    int m_event_index = m_tid;
    int m_stride = NUM_GNOM_PRE_THREADS;

    int batch_launch_count = 0;

    checkCudaErrors(cuCtxPushCurrent(m_wta->cuda_ctx->hcuContext));

    ngd_req *m_gnom_ngd_req;
    gpu_nom_req *gnr;
    gpu_stream_pair m_gsp;

    CUevent *m_event = NULL;

    pthread_mutex_lock( &csm->m_host_mutex );
    printf("NGD GNoM_pre %d\n", m_tid);
    print_index((const char*)"\tStream_index", m_tid, m_stream_index, m_stride);
    print_index((const char*)"\tEvent_index", m_tid, m_event_index, m_stride);

    pthread_mutex_unlock( &csm->m_host_mutex );

    int threadsPerBlock = 512;
    int blocksPerGrid = 0;
    if(NUM_REQUESTS_PER_BATCH <  512){
        // threadsPerBlock = NUM_REQUESTS_PER_BATCH; // Maximum number of threads per block
    }

    blocksPerGrid = NUM_REQUESTS_PER_BATCH / 256; // 256 requests per block, 512 threads per block

    printf("threadsPerBlock=%d, blocksPerGrid=%d\n", threadsPerBlock, blocksPerGrid);

    // Kernel Arguments. Fill in NULL arguments at runtime
#ifdef NETWORK_ONLY_TEST
    void *cuda_args[] = { NULL, NULL, NULL, NULL};
#else
    // Last two arguments for DEBUG
    void *cuda_args[] = { NULL, NULL, NULL, (void *)&hashpower,  (void *)&csm->m_gpu_hashtable_ptr, (void *)&csm->m_gpu_lock_ptr, NULL /* , NULL, NULL */};
#endif
    rel_time_t timestamp = 0;


#ifdef BIND_CORE
    set_thread_affinity(m_tid);
    //set_thread_affinity(0);
#endif

    unsigned long long num_batches = 0;
    double total_time = 0.0;
    struct timespec start_t, end_t;

    int req_batch_size = 0;

    int continue_count = 0;
    clock_gettime(CLOCK_REALTIME, &start_t);
    do{

        // Read request batch from PF_RING
        if(unlikely(!csm->gnom_ngd_read(&m_gnom_ngd_req, m_tid)) ){
            cout << "Error: No data returned from NGD read... Exiting" << endl;
            break;
        }
        
        if(unlikely(stop_stream_manager)) break;

        gnr = m_gnom_ngd_req->m_gnom_req;

        if(unlikely(gnr->in_use)){
            printf("ERROR: Ran out of requests...(batch %llu, cur_index: %d)\n", num_batches, m_stream_index);
            continue_count++;
            fflush(stdout);
            continue;
        }

        gnr->in_use = 1;

        gnr->extra = (void *)m_gnom_ngd_req; // Used for recycling NGD to free list later
        req_batch_size = m_gnom_ngd_req->tot_size;

        // Configure GNR request
        gnr->req_ptr = (size_t *)m_gnom_ngd_req->h_ptr;                     // Host request ptr
        gnr->gpu_req_ptr = m_gnom_ngd_req->d_ptr;                           // GPU request ptr

        gnr->res_ptr = (size_t *)m_gnom_ngd_req->h_res_ptr;
        gnr->gpu_res_ptr = m_gnom_ngd_req->d_res_ptr;

        gnr->num_req = m_gnom_ngd_req->n_req;

        // Set stream
        gnr->queue_id = m_stream_index;

        gnr->timestamp = current_time;
        gnr->req_id = global_req_id++;
        gnr->batch_id = 0;
        gnr->queue_ind = 0;

        num_batches++;

//        printf("req_ptr: %p, greq_ptr: %p, res_ptr: %p, gres_ptr: %p, num_req: %d, q_id: %d, timestamp: %d, req_id: %d, batch_id: %d, queue_ind: %d\n",
//                gnr->req_ptr, gnr->gpu_req_ptr, gnr->res_ptr, gnr->gpu_res_ptr, gnr->num_req, gnr->queue_id, gnr->timestamp, gnr->req_id, gnr->batch_id, gnr->queue_ind);


        // Populate arguments
        cuda_args[0] = (void *)&gnr->gpu_req_ptr;
        cuda_args[1] = (void *)&gnr->num_req;
        cuda_args[2] = (void *)&gnr->gpu_res_ptr;

#ifdef NETWORK_ONLY_TEST
        cuda_args[3] = (void *)&gnr->timestamp;
#else
        cuda_args[6] = (void *)&gnr->timestamp;
#endif


#ifndef USE_DIRECT_ACCESS_MEM_REQUEST
        checkCudaErrors(cuMemcpyHtoDAsync_v2(gnr->gpu_req_ptr,
                                             gnr->req_ptr,
                                             req_batch_size,
                                             csm->m_streams[m_stream_index]));
#endif

        // GNoM - Start the timer
        clock_gettime(CLOCK_REALTIME, &gnr->start_time);



        // Launch kernel
        // Data has already been copied to the GPU directly via GPUdirect. Launch kernel.
#ifdef NETWORK_ONLY_TEST
        checkCudaErrors(cuLaunchKernel(m_wta->cuda_ctx->network_function, blocksPerGrid, 1, 1,
                                       threadsPerBlock, 1, 1, 0,
                                       csm->m_streams[m_stream_index], cuda_args, NULL));
#else
        checkCudaErrors(cuLaunchKernel(m_wta->cuda_ctx->hcuFunction, blocksPerGrid, 1, 1,
                                       threadsPerBlock, 1, 1, 0,
                                       csm->m_streams[m_stream_index], cuda_args, NULL));
#endif


        // Copy Responses
#ifndef USE_DIRECT_ACCESS_MEM_RESPONSE
        // Copy response buffer back from GPU
        checkCudaErrors(cuMemcpyDtoHAsync_v2(gnr->res_ptr, gnr->gpu_res_ptr, res_buf_size, csm->m_streams[m_stream_index]));
#endif

        // Record event to signal the completion of this kernel
        checkCudaErrors(cuEventRecord(csm->m_gpu_event_queue[m_event_index], csm->m_streams[m_stream_index]));
        
        m_gsp.gpu_event = &csm->m_gpu_event_queue[m_event_index];
        m_gsp.gnr = gnr;

        // Push event into map
        pthread_mutex_lock(&csm->m_map_mutex[m_stream_index]);
        csm->m_k_event_queue[m_stream_index]->push(m_gsp);
        pthread_mutex_unlock(&csm->m_map_mutex[m_stream_index]);

        m_stream_index = (m_stream_index + m_stride) % NUM_C_KERNELS;
        m_event_index = (m_event_index + m_stride) % MAX_NUM_BATCH_INFLIGHT;

    }while(likely(!stop_stream_manager));

    pthread_mutex_lock( &csm->m_host_mutex );
    csm->num_batches_processed += num_batches;
    printf("GNoM-Pre %d: processed %llu batches (continue_count = %d)\n", m_tid, num_batches, continue_count);
    fflush(stdout);
    pthread_mutex_unlock( &csm->m_host_mutex );


    while(csm->num_gpu_nom_req_in_flight > 0){
        usleep(2);
    }
    clock_gettime(CLOCK_REALTIME, &end_t);
    total_time = (double)((end_t.tv_sec - start_t.tv_sec) + ((end_t.tv_nsec - start_t.tv_nsec)/1E9));

    stop_stream_manager = true;

    printf("Cleaning up GNoM-NGD framework\n");

    sleep(1);
    if(m_tid == 0){ // Only the first thread should clean things up
        csm->stop_gpu_nom();
        sleep(3);
        abort();
    }

    sleep(2);

    return NULL;
}

void *cuda_stream_manager_ngd::gnom_post(void *arg){
    worker_thread_args *m_wta = (worker_thread_args *)arg;
    cuda_stream_manager_ngd *csm = (cuda_stream_manager_ngd *)m_wta->csm;
    int m_tid = m_wta->tid;
    size_t *RXbp = m_wta->mmaped_RX_buffer_pointers;
    size_t *TXbp = m_wta->mmaped_TX_buffer_pointers;
 
    printf("NGD GNoM_post\n");

    checkCudaErrors(cuCtxPushCurrent(m_wta->cuda_ctx->hcuContext));

    gnom_buf_recycle_info recycle_info;

    CUresult status;
    CUevent *event;

    int m_stream_ind = m_tid;
    int m_stream_ind_stride = NUM_GNOM_POST_THREADS;

    int wt_batch_cnt = 0;
    int ret = 0;
    int res_cnt = 0;

#ifdef NETWORK_ONLY_TEST
    int hdr_size = 42;
#else
    // UDP Packet header (42 bytes) + 8 Byte Memcached header = 50 Bytes.
    int hdr_size = 50;
#endif

    int buffer_idx = 0;
    int flush_packet = 0;
    int sent_bytes = 0;

    size_t *item_ptr = NULL;
    u_char *pkt_hdr_ptr = NULL;
    u_char *pf_pkt_buffer_ptr = NULL;
    item *itm = NULL;

    pfring_zc_queue *pf_zq = NULL;
    pfring_zc_pkt_buff *pf_buf = NULL;
    pfring_zc_pkt_buff **m_pf_bufs = NULL;


    gpu_stream_pair m_gsp;
    gpu_nom_req *m_req;
    ngd_req *m_ngd_req;

    unsigned long long num_memc_misses = 0;
    unsigned long long num_memc_hits = 0;
    uint64_t num_pkt_processed = 0;

    unsigned start_pf_buffer_ind = 0;
    unsigned cur_pf_buffer_ind = 0;
    unsigned free_pf_buffer_ind = 0;

#ifdef BIND_CORE
    //    set_thread_affinity((m_tid+1) % 4); //
    //set_thread_affinity(m_tid + 2);
#endif

    pf_zq = csm->m_pf_zq[m_tid];
    m_pf_bufs = csm->m_pf_buffers[m_tid]; // Point to correct PF_RING buffers for this thread


    pthread_mutex_lock( &csm->m_host_mutex );
    printf("Puller thread: %d | start_index: %d | stride: %d \n", m_tid, m_stream_ind, m_stream_ind_stride);
    pthread_mutex_unlock( &csm->m_host_mutex );


    // Each worker thread looks at a stride 2 of queues
    //      Thread 0: 0, 2, 4, 6, 8...
    //      Thread 1: 1, 3, 5, 7, 9...
    // - Previously each thread took a subset of work. This didn't work because there are only ever 1-3 things in the queue at a time.
    //   This meant only one thread was ever really doing anything at a time, and then eventually just passed the work off to the next thread.
    // - And now in 4 stages.
    //      - (0) Check stream for completed kernel (no mutex, each thread has own set of streams in a strided fashion)
    //      - (1) Populate PF_RING buffers (No mutex, each thread has own set of buffers)
    //      - (2) Send to PF_RING (mutex 2)
    //      - (3) Recycle GRX buffers (mutex 1)
    do{

        // Kernels pushed in FCFS, check for first completed on a per-command queue basis
        //pthread_mutex_lock(&csm->m_map_mutex[m_stream_ind]);
        if(!csm->m_k_event_queue[m_stream_ind]->empty()){ // Something in the stream queue

            // (0)
            m_gsp = csm->m_k_event_queue[m_stream_ind]->front();
            status = cuEventQuery(*m_gsp.gpu_event);

            if(status == CUDA_SUCCESS){ // Stream operations have completed for this stream

                csm->m_k_event_queue[m_stream_ind]->pop();

                m_req = m_gsp.gnr;
                m_ngd_req = (ngd_req *)m_req->extra;

                // Perform response
                item_ptr = (size_t *)m_req->res_ptr;
                pkt_hdr_ptr = (u_char *)((size_t)m_req->res_ptr + NUM_REQUESTS_PER_BATCH*sizeof(size_t)); // Pkt headers are passed the item pointers in the response memory


                // (1)
                start_pf_buffer_ind = free_pf_buffer_ind;
                cur_pf_buffer_ind = free_pf_buffer_ind;
                free_pf_buffer_ind = (free_pf_buffer_ind+NUM_REQUESTS_PER_BATCH) % NUM_PF_BUFS;

                for(unsigned i=0; i<NUM_REQUESTS_PER_BATCH; ++i){
                    pf_buf = m_pf_bufs[cur_pf_buffer_ind];
                    cur_pf_buffer_ind++;

                    // Set pf_ring buffer pointer
                    pf_pkt_buffer_ptr = pf_buf->data;

                    // Set buffer length
                    pf_buf->len = RESPONSE_SIZE; // FIXME: Use actual buffer length based off of item size

#ifdef NETWORK_ONLY_TEST
#ifdef LATENCY_MEASURE
                    // If network only test, no payload, just send packet
                    memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, hdr_size+8); // hdr + 8 byte latency timestamp from client
                    pf_pkt_buffer_ptr+=(hdr_size+8);
#else
                    // If network only test, no payload, just send packet
                    memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, hdr_size);
                    pf_pkt_buffer_ptr+=hdr_size;

#endif
#else
                    if(item_ptr[i] != 0){ // If an item was found for this request
                        num_memc_hits++;
                        itm = (item *)item_ptr[i];

                        // Copy packet header + VALUE + key in one from GPU
#ifdef LATENCY_MEASURE
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 52);
                        pf_pkt_buffer_ptr+= (52);
#else
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 48);
                        pf_pkt_buffer_ptr+= (48);
#endif

                        memcpy(pf_pkt_buffer_ptr, ITEM_key(itm), itm->nkey);
                        pf_pkt_buffer_ptr+= (itm->nkey);

                        // Copy Suffix + Value
                        memcpy(pf_pkt_buffer_ptr, ITEM_suffix(itm), itm->nsuffix + itm->nbytes);
                        pf_pkt_buffer_ptr += (itm->nsuffix + itm->nbytes);

                    }else{ // else, no item was found
                        num_memc_misses++;
                        // Copy packet header + VALUE
#ifdef LATENCY_MEASURE
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 52);
                        pf_pkt_buffer_ptr+= (52);
#else
                        memcpy(pf_pkt_buffer_ptr, pkt_hdr_ptr, 48);
                        pf_pkt_buffer_ptr+= (48);
#endif

                        // TODO: Move this to GPU too
                        memcpy(pf_pkt_buffer_ptr, NF, strlen(NF));
                        pf_pkt_buffer_ptr += strlen(NF);
                    }
                    //}
                    // Write "END"
                    memcpy(pf_pkt_buffer_ptr, END, strlen(END));
#endif // End NETWORK_ONLY

                    pkt_hdr_ptr += RESPONSE_HDR_STRIDE; // Now using 256 Bytes per response packet stride on GPU - Move to next response packet
                }

                // (2)
                cur_pf_buffer_ind = start_pf_buffer_ind; // Move back to first PF_RING buffer in this batch
                //pthread_mutex_lock(&csm->pf_ring_send_mutex); // Lock PF_RING TX
                for(unsigned i=0; i<NUM_REQUESTS_PER_BATCH; ++i){
                    pf_buf = m_pf_bufs[cur_pf_buffer_ind];
                    cur_pf_buffer_ind++;

#if 0
                    ip_header *iph = (ip_header *)((size_t)pf_buf->data + sizeof(ether_header));
                    if(iph->daddr == htonl(0xC0A80007))
                        csm->print_pkt_hdr((void *)pf_buf->data);
#endif

                    // Push the packets out the door
                    while( (sent_bytes = pfring_zc_send_pkt(pf_zq, &pf_buf, flush_packet)) < 0 ){
                        if(unlikely(stop_stream_manager))
                            break;
                    }
                }
                num_pkt_processed += NUM_REQUESTS_PER_BATCH;
                pfring_zc_sync_queue(pf_zq, tx_only); // Sync after a batch is sent
                //pthread_mutex_unlock(&csm->pf_ring_send_mutex); // Lock PF_RING TX

                wt_batch_cnt++;

                // GNoM: Ensure ordering of GET/SET requests on GPU is preserved on the CPU
                csm->m_get_complete_timestamp[m_tid] = m_req->timestamp;

                m_req->in_use = 0;

                // (3)
                csm->gnom_ngd_write(m_ngd_req);


            }else{
                //usleep(1); // Wait 5 uS before checking again... Avoid constant driver calls
                //pthread_mutex_unlock(&csm->m_map_mutex[m_stream_ind]);
            }    
        }else{
            //pthread_mutex_unlock(&csm->m_map_mutex[m_stream_ind]);
        }

        // Check the next stream allocated to this puller thread
        m_stream_ind = (m_stream_ind + m_stream_ind_stride) % (NUM_C_KERNELS);

    }while(!stop_stream_manager);

    pthread_mutex_lock( &csm->m_host_mutex );
    total_num_pkt_processed += num_pkt_processed;
    csm->num_memc_misses += num_memc_misses;
    csm->num_memc_hits += num_memc_hits;
    stop_stream_manager = true;
    printf("Puller thread %d complete...\n", m_tid);
    pthread_mutex_unlock( &csm->m_host_mutex );


    return NULL;
}

void cuda_stream_manager_ngd::stop_gpu_nom(){
    static int gnom_cleanup_ngd_called = 0;

    if(gnom_cleanup_ngd_called == 0){
        gnom_cleanup_ngd_called = 1;
        print_stats(); // Print stats

        cout << "Shutting down GNoM NGD..." << endl;

#ifndef USE_DIRECT_ACCESS_MEM_REQUEST
        if(m_kernel_arg_hdr.buf_meta_data.cpu_args->cuda_page_va){
            checkCudaErrors(cuMemFree(m_kernel_arg_hdr.buf_meta_data.cpu_args->cuda_page_va));
        }

        if(m_kernel_arg_hdr.buf_meta_data.cpu_args->user_page_va){
            free(m_kernel_arg_hdr.buf_meta_data.cpu_args->user_page_va);
        }
#else
        printf("Error: zero-copy memory not implemented with the NGD configuration\n");
#endif
        // Destroy all CUDA event objects
        for(unsigned i=0; i<MAX_NUM_BATCH_INFLIGHT; ++i){
            cuEventDestroy(m_gpu_event_queue[i]);
        }

        if(m_kernel_arg_hdr.buf_meta_data.cpu_args)
            free(m_kernel_arg_hdr.buf_meta_data.cpu_args);


        //for(unsigned i=0; i<m_n_batches; ++i){
        while(!m_ngd_free_reqs.empty()){
            ngd_req *ngdr = m_ngd_free_reqs.front();
            //m_ngd_free_reqs.pop();
            m_ngd_free_reqs.pop_front();
            delete ngdr;
        }

        // Freeing SET request buffers
         if(d_set_req)
             checkCudaErrors(cuMemFree(d_set_req));

         if(d_set_res)
             checkCudaErrors(cuMemFree(d_set_res));


         for(unsigned i=0; i<MAX_NUM_BATCH_INFLIGHT; ++i){
#ifdef USE_DIRECT_ACCESS_MEM_RESPONSE
             if(host_response_buffers[i])
                 cuMemFreeHost(host_response_buffers[i]);
#else
             if(host_response_buffers[i])
                 free(host_response_buffers[i]);

             if(gpu_nom_response_buffers[i])
                 cuMemFree_v2(gpu_nom_response_buffers[i]);
#endif
         }

        if(m_gpu_nom_req)
            delete m_gpu_nom_req;

        if(gpu_nom_response_buffers)
            delete gpu_nom_response_buffers;

        if(host_response_buffers)
            delete host_response_buffers;

        if(m_gpu_event_queue)
            delete m_gpu_event_queue;

        cout << "GNoM NGD destroyed successfully" << endl;
    }
}

bool cuda_stream_manager_ngd::gnom_init_pf_ring_rx(int cluster_id){
    int num_queue_buffers = 32768;

    // Each open a different Queue on the device. Creating up to 8 RX queues, currently only using 1 (eth2@0)
    const char *pf_devices_rx[MAX_PF_RING_RX] = {"zc:eth2@0", "zc:eth2@1", "zc:eth2@2", "zc:eth2@3",
        "zc:eth2@4", "zc:eth2@5", "zc:eth2@6", "zc:eth2@7"};

    m_pf_cluster = pfring_zc_create_cluster(cluster_id,
                                            1536,
                                            0,
                                            MULTI_PF_RING_RX * (num_queue_buffers + NUM_PF_BUFS),
                                            numa_node_of_cpu(0),
                                            /*0,*/
                                            NULL);
    if(m_pf_cluster == NULL) {
        fprintf(stderr, "pfring_zc_create_cluster error [%s] Please check your hugetlb configuration\n", strerror(errno));
            return false;
    }
    printf("pfring_zc_create_cluster success...\n");
        
    m_pf_zq_rx[0] = pfring_zc_open_device(m_pf_cluster, pf_devices_rx[0], rx_only, 0);

    if(m_pf_zq_rx[0] == NULL) {
        fprintf(stderr, "pfring_zc_open_device %d error [%s] Please check that %s is up and not already used\n",
                0, strerror(errno), pf_devices_rx[0]);
        return false;
    }


    for(unsigned i=0; i<MULTI_PF_RING_RX; ++i){
        m_pf_buf_ind[i] = 0;

#if 0
        // Look at this code if using more than one GNoM-Pre thread to receive from PF_RING and push to the GPU.
        m_pf_zc_rx[i] = pfring_zc_create_cluster(cluster_id + i,
                                              1536,
                                              0,
                                              num_queue_buffers + NUM_PF_BUFS,
                                              numa_node_of_cpu(i % 4),
                                              /*0,*/
                                              NULL);


        if(m_pf_zc_rx[i] == NULL) {
            fprintf(stderr, "pfring_zc_create_cluster %d error [%s] Please check your hugetlb configuration\n", i, strerror(errno));
            return false;
        }
        printf("pfring_zc_create_cluster %d success...\n", i);

#endif

        for (unsigned j = 0; j < NUM_PF_BUFS; j++) {
            //m_pf_buffers_rx[i][j] = pfring_zc_get_packet_handle(m_pf_zc_rx[i]);
            m_pf_buffers_rx[i][j] = pfring_zc_get_packet_handle(m_pf_cluster);
            if (m_pf_buffers_rx[i][j] == NULL) {
                fprintf(stderr, "pfring_zc_get_packet_handle error\n");
                return false;
            }
        }
        printf("pfring_zc_get_packet_handle %d success...\n", i);

#if 0        
        m_pf_zq_rx[i] = pfring_zc_open_device(m_pf_cluster, pf_devices_rx[i], rx_only, 0);

        if(m_pf_zq_rx[i] == NULL) {
            fprintf(stderr, "pfring_zc_open_device %d error [%s] Please check that %s is up and not already used\n",
                    i, strerror(errno), pf_devices_rx[i]);
            return false;
        }

       
#endif
    }

    fprintf(stderr, "PF_RING receiving packets through %s\n\n", pf_devices_rx[0]);

    return true;
}

//////////////////////////////////////////////////////////////////////////////////
/////////////////// Various Helper Functions and test routines ///////////////////
//////////////////////////////////////////////////////////////////////////////////

/************** Segfault handlers for debugging **************/
// Simple segfault handler, catch and print out any information about the segfault
void segfault_handler(int signal, siginfo_t *info, void *arg){
    ucontext_t *context = (ucontext_t *)arg;
    fprintf(stderr,
        "si_signo:  %d\n"
        "si_code:   %s\n"
        "si_errno:  %d\n"
        "si_pid:    %d\n"
        "si_uid:    %d\n"
        "si_addr:   %p\n"   /* Memory address that caused the segfault */
        "si_status: %d\n"
        "si_band:   %ld\n",
        info->si_signo,
        (info->si_code == SEGV_MAPERR) ? "SEGV_MAPERR" : "SEGV_ACCERR",
        info->si_errno, info->si_pid, info->si_uid, info->si_addr,
        info->si_status, info->si_band
    );

    fprintf(stderr,
        "uc_flags:  0x%lx\n"
        "ss_sp:     %p\n"
        "ss_size:   %ld\n"
        "ss_flags:  0x%X\n",
        context->uc_flags,
        context->uc_stack.ss_sp,
        context->uc_stack.ss_size,
        context->uc_stack.ss_flags
    );

    fprintf(stderr, "General Registers:\n");
    for(int i = 0; i < NGREG; i++)
        fprintf(stderr, "\t%7s: 0x%llx\n", gregs[i], context->uc_mcontext.gregs[i]);

    cuda_stream_manager_base::stop_stream_manager = true;

    global_csm->stop_gpu_nom();

    exit(-1);
}

void sig_handler(int sig) {
  static int called = 0;
  fprintf(stderr, "CTRL+C, calling stop...\n");
  if(called) return; else called = 1;

  // Set STOP flag, main GNoM_pre thread will cleanup
  cuda_stream_manager_base::stop_stream_manager = true;

}

int set_thread_affinity(unsigned core){
    int ret = 0;
    cpu_set_t cs;

    CPU_ZERO(&cs);
    CPU_SET(core, &cs);

    ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cs);
    if(ret != 0){
        printf("Error: Could not set thread affinity to core %d (%d)", core, ret);
        return -1;
    }

    return 0;
}


void print_index(const char *string, int tid, int start, int end){
    cout << string << " <" << tid << "> (" << start << "," << end << ")" << endl;
    return;
}

void inc_wrap_index(int &ind, int start, int end){
    ind++;
    if(ind >= end)
        ind = start;

    return;
}


int in_cksum(unsigned char *buf, unsigned nbytes, int sum) {
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    sum += (u_int16_t) ntohs(*((u_int16_t *)(buf + i)));
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}

static u_int32_t wrapsum (u_int32_t sum) {
  sum = ~sum & 0xFFFF;
  return htons(sum);
}

void test_mapping_cuda_buffer(CUdeviceptr dev_ptr, size_t *host_ptr, int gnom_handle){
    int res = 0;


    cout << "Mapping page..." << endl;

    res = ioctl(gnom_handle, TEST_MAP_SINGLE_PAGE, &host_ptr);
    if(res != 0){
        cout <<"ioctl_err: abort()" << endl;
    }


    cout << "Setting packet buffer..." << endl;

    int pkt_size = 54; // 42 byte header + 12 byte payload
    u_int32_t src_ip = 0xC0A80002 /* from 192.168.0.2 */;
    u_int32_t dst_ip =  0xC0A80104 /* 192.168.1.4 */;

    ether_header *eh = (ether_header *)host_ptr;
    ip_header *iph = (ip_header *)((size_t)host_ptr + sizeof(ether_header));
    udp_header *uh = (udp_header *)((size_t)host_ptr + sizeof(ether_header) + sizeof(ip_header));
    const char *payload = (const char *)((size_t)host_ptr + sizeof(ether_header) + sizeof(ip_header) + sizeof(udp_header));

    // Set packet buffer contents
    // Ether
    eh->ether_type = htons(0x0800);
    eh->ether_shost[0] = 0x68;
    eh->ether_shost[1] = 0x05;
    eh->ether_shost[2] = 0xCA;
    eh->ether_shost[3] = 0x13;
    eh->ether_shost[4] = 0xCE;
    eh->ether_shost[5] = 0x79;
    eh->ether_dhost[0] = 0x68;
    eh->ether_dhost[1] = 0x05;
    eh->ether_dhost[2] = 0xCA;
    eh->ether_dhost[3] = 0x1B;
    eh->ether_dhost[4] = 0x1E;
    eh->ether_dhost[5] = 0x66;

    // IP
    iph->ihl = 5;
    iph->version = 4;
    iph->tos = 0;
    iph->tot_len = htons(pkt_size - sizeof(ether_header));
    iph->id = htons(9930);
    iph->ttl = 64;
    iph->frag_off = htons(0);
    iph->protocol = IPPROTO_UDP;
    iph->daddr = htonl(dst_ip);
    iph->saddr = htonl(src_ip);
    iph->check = 0;
    iph->check = wrapsum(in_cksum((unsigned char *)iph, 4*iph->ihl, 0));

    // UDP
    uh->source = htons(7777);
    uh->dest = htons(7778);
    uh->len = htons(pkt_size - sizeof(ether_header) - sizeof(ip_header));
    uh->check = 0;

    payload = (const char *)"test\n";

    cout << "Testing send..." << endl;
    res = ioctl(gnom_handle, TEST_SEND_SINGLE_PACKET, &host_ptr);
    if(res != 0){
        cout <<"ioctl_err: abort()" << endl;
    }


    cout << "unmapping page..." << endl;
    res = ioctl(gnom_handle, TEST_UNMAP_SINGLE_PAGE, &host_ptr);
    if(res != 0){
        cout <<"ioctl_err: abort()" << endl;
    }

    return;

}

int verify_pkt(void *data){
    udp_header *uh = (udp_header *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
    if(ntohs(uh->source) == 9960){
        return 1;
    }else{
        return 0;
    }
}

void test_spoof_packet(size_t *data){
    int res = 0;

    int pkt_size = PKT_STRIDE; // 42 byte header + 12 byte payload
    u_int32_t src_ip = 0xC0A80002 /* from 192.168.0.2 */;
    u_int32_t dst_ip =  0xC0A80104 /* 192.168.1.4 */;

    ether_header *eh = (ether_header *)data;
    ip_header *iph = (ip_header *)((size_t)data + sizeof(ether_header));
    udp_header *uh = (udp_header *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
    const char *payload = (const char *)((size_t)data + sizeof(ether_header) + sizeof(ip_header) + sizeof(udp_header));

    // Set packet buffer contents
    // Ether
    eh->ether_type = htons(0x0800);
    eh->ether_shost[0] = 0x68;
    eh->ether_shost[1] = 0x05;
    eh->ether_shost[2] = 0xCA;
    eh->ether_shost[3] = 0x13;
    eh->ether_shost[4] = 0xCE;
    eh->ether_shost[5] = 0x79;
    eh->ether_dhost[0] = 0x68;
    eh->ether_dhost[1] = 0x05;
    eh->ether_dhost[2] = 0xCA;
    eh->ether_dhost[3] = 0x1B;
    eh->ether_dhost[4] = 0x1E;
    eh->ether_dhost[5] = 0x66;

    // IP
    iph->ihl = 5;
    iph->version = 4;
    iph->tos = 0;
    iph->tot_len = htons(pkt_size - sizeof(ether_header));
    iph->id = htons(9930);
    iph->ttl = 64;
    iph->frag_off = htons(0);
    iph->protocol = IPPROTO_UDP;
    iph->daddr = htonl(dst_ip);
    iph->saddr = htonl(src_ip);
    iph->check = 0;
    iph->check = wrapsum(in_cksum((unsigned char *)iph, 4*iph->ihl, 0));

    // UDP
    uh->source = htons(9191);
    uh->dest = htons(9960);
    uh->len = htons(pkt_size - sizeof(ether_header) - sizeof(ip_header));
    uh->check = 0;

   // payload = (const char *)"test\n";


    return;

}


