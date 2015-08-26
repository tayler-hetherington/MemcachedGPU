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
 * cuda_network.cu
 */

// CUDA utilities and system includes
#ifndef __CUDA_VERSION__
#endif

#include <cuda_runtime.h>
#include <host_defines.h>
#include <device_launch_parameters.h>

#include <stdio.h>


//#define DEBUG_PRINT

// Tacks on an 8B timestamp from the client to the response packet header
#define LATENCY_MEASURE

#define RESPONSE_HDR_STRIDE 256

/*************************************/
#define REQUEST_GROUP_SIZE      128 // Do not change // Number of requests per group (subset of batch)
#define MAX_THREADS_PER_BLOCK   256 // Number of threads per request group
/*************************************/
#define NUM_REQUESTS_PER_BATCH   512 // Match with gpu_common.h and gpu_km_shared.h
#define NUM_REQUESTS_PER_GROUP   256 // Do not change

#define NUM_THREADS_PER_GROUP   NUM_REQUESTS_PER_BATCH*2
#define NUM_GROUPS NUM_REQUESTS_PER_BATCH / NUM_REQUESTS_PER_GROUP
/*************************************/

#define SINGLE_GPU_PKT_PTR
#define RX_BUFFER_SZ 2048

#define UDP_PORT        9960
#define ETH_ALEN        6
#define IPPROTO_UDP     17


// Constant response packet size (just send a small response packet to the client)
#define RESPONSE_SIZE   80

#define G_HTONS(val) (u_int16_t) ((((u_int16_t)val >> 8) & 0x00FF ) | (((u_int16_t)val << 8) & 0xFF00) )
#define G_NTOHS(val) (G_HTONS(val))

typedef unsigned int rel_time_t;

/*********************************************/
// Network Packet Headers
/*********************************************/
typedef struct _ether_header{
  u_int8_t  ether_dhost[ETH_ALEN];  /* destination eth addr */
  u_int8_t  ether_shost[ETH_ALEN];  /* source ether addr    */
  u_int16_t ether_type;             /* packet type ID field */
}ether_header;

typedef struct _ip_header {
  u_int8_t  version;                /* version */           // Version+ihl = 8 bits, so replace ihl with 8bit version
  //u_int32_t ihl:4;                /* header length */

  u_int8_t  tos;                    /* type of service */
  u_int16_t tot_len;                /* total length */
  u_int16_t id;                     /* identification */
  u_int16_t frag_off;               /* fragment offset field */
  u_int8_t  ttl;                    /* time to live */
  u_int8_t  protocol;               /* protocol */
  u_int16_t check;                  /* checksum */

  u_int16_t saddr1;                 /* Break source and dest address into */
  u_int16_t saddr2;                 /* 16-bits to avoid alignment issues*/
  u_int16_t daddr1;
  u_int16_t daddr2;

}ip_header;

typedef struct _udp_header {
  u_int16_t source;     /* source port */
  u_int16_t dest;       /* destination port */
  u_int16_t len;        /* udp length */
  u_int16_t check;      /* udp checksum */
}udp_header;
/*********************************************/
/*********************************************/

/*********************************************/
// GNoM internal network data structures
/*********************************************/
typedef struct _pkt_memc_hdr_{
    ether_header eh;
    ip_header iph;
    udp_header udp;
}pkt_memc_hdr;

typedef struct _pkt_res_memc_hdr_{
    ether_header eh;
    ip_header iph;
    udp_header udp;
}pkt_res_memc_hdr;

typedef struct _mod_pkt_info_{
    int is_get_req;
    pkt_memc_hdr nmch; // Packet header + memc 8 Byte header
}mod_pkt_info;

/*********************************************/
/*********************************************/


__device__ void print_pkt_hdr(void *data){
    unsigned i=0;
    ether_header *eh = (ether_header *)data;
    ip_header *iph = (ip_header *)((size_t)data + sizeof(ether_header));
    udp_header *uh = NULL;

    printf("Packet header contents: \n");

    /***** ETHERNET HEADER *****/
    printf("\t==Ethernet header==\n");
    printf("\t\tDest: ");
    for(i=0; i<ETH_ALEN; ++i)
        printf("%hx ", eh->ether_dhost[i]);
    printf("\n\t\tSource: ");
    for(i=0; i<ETH_ALEN; ++i)
        printf("%hx ", eh->ether_shost[i]);
    printf("\n\t\tType: %hx\n", eh->ether_type);
    /***** END ETHERNET HEADER *****/

    /***** IP HEADER *****/
    printf("\t==IP header==\n");
    printf("\t\tVersion+hdr_len: %hu\n", iph->version);
    printf("\t\tTOS: %hu\n", iph->tos);
    printf("\t\tTotal Length: %hu\n", G_NTOHS(iph->tot_len));
    printf("\t\tID: %hu\n", G_NTOHS(iph->id));
    printf("\t\tFrag_off: %hu\n", iph->frag_off);
    printf("\t\tTTL: %hu\n", iph->ttl);
    printf("\t\tProtocol: %hu\n", iph->protocol);
    printf("\t\tchecksum: %hu\n", G_NTOHS(iph->check));
    printf("\t\tSource address: %x\n", (iph->saddr1 << 16) | iph->saddr2);
    printf("\t\tDest address: %x\n", (iph->daddr1 << 16) | iph->daddr2);
    /***** END IP HEADER *****/


    /***** UDP HEADER *****/
    uh = (udp_header *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
    printf("\t==UDP header==\n");
    printf("\t\tSource port: %hu\n", G_NTOHS(uh->source));
    printf("\t\tDest port: %hu\n", G_NTOHS(uh->dest));
    printf("\t\tLength: %hu\n", G_NTOHS(uh->len));
    printf("\t\tChecksum: %hu\n", uh->check);
    /***** END UDP HEADER *****/



}

__device__ int in_cksum(unsigned char *buf, unsigned nbytes, int sum) {
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + i)));
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

/* ******************************************* */

__device__ int wrapsum (u_int32_t sum) {
  sum = ~sum & 0xFFFF;
  return G_NTOHS(sum);
}

/* ******************************************* */



/* ******************************************* */
// Forward declarations
__device__ void parse_pkt(unsigned long long first_RX_buffer_ptr, int local_tid, int logical_tid, int thread_type,  mod_pkt_info *mpi);
__device__ void create_response_header(mod_pkt_info *mpi, int helper_tid);
__device__ void populate_response(size_t *res_mem, mod_pkt_info *mpi, int tid, int helper_tid, int group_id, int *item_is_found, unsigned thread_type, int cta_id);
__device__ void new_populate_response(long long *TX_buf_ptrs, int local_tid, mod_pkt_info *mpi);
/* ******************************************* */

extern "C" __global__ void network_kernel(unsigned long long first_req_addr,        // Address of first CUDA buffer
                                            int num_req,                    // # of requests
                                            int *response_mem,              // Memory allocated for responses
                                            rel_time_t timestamp){


    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    int thread_type = ((local_tid % MAX_THREADS_PER_BLOCK) < (MAX_THREADS_PER_BLOCK / 2)) ? 0 : 1;// 0 means actual request threads, 1 means helper threads

    int group_id;

    if(local_tid < MAX_THREADS_PER_BLOCK){
        group_id = 0;
    }else{
        group_id = 1;
    }

    int half_group_size = MAX_THREADS_PER_BLOCK/2; // ==> 256/2 = 128

    int logical_tid = -1;
    if(thread_type == 0){
        logical_tid = (group_id * half_group_size) + (tid % half_group_size); // First half looks
    }else{
        logical_tid = (group_id * half_group_size) + ( (tid-half_group_size) % half_group_size);
    }

    // Testing
    __shared__ mod_pkt_info mpi[NUM_REQUESTS_PER_GROUP];
    __shared__ int item_is_found[NUM_REQUESTS_PER_GROUP];

    // First Thread Block looks at start address (first_req_addr).
    // Second Thread Block looks at start address of buffer 256 == first_req_addr + 256 req * 2048 B/req
    unsigned long long m_first_RX_buffer_addr = first_req_addr + (blockIdx.x*RX_BUFFER_SZ*NUM_REQUESTS_PER_GROUP);

#ifdef DEBUG_PRINT
    if(tid == 0){
        print_pkt_hdr((void *)m_req_buf_ptrs[0]);
    }
#endif

    parse_pkt(m_first_RX_buffer_addr, local_tid, logical_tid, thread_type, mpi);

    __syncthreads();

    if(thread_type == 0){
        create_response_header(&mpi[logical_tid], logical_tid);
    }

    __syncthreads();

    populate_response((size_t *)response_mem, mpi, local_tid, logical_tid, group_id, item_is_found, thread_type, blockIdx.x);

    __syncthreads();
    __threadfence_system();


}


// TODO: Currently hardcoded to 256 requests per subgroup. This likely doesn't need to
//       change unless trying to pack more requests per thread/group.
#define NUM_REQ_PER_LOOP    16
#define WARP_SIZE           32
#define THREAD_PER_HDR_COPY 13  // 13 threads * 4 bytes = 52 bytes / hdr


__device__ void new_populate_response(long long *TX_buf_ptrs, int local_tid,
                                      mod_pkt_info *mpi){

    int *res_ptr = NULL;
    int *pkt_hdr_ptr = NULL;


    int req_ind = (int)(local_tid / WARP_SIZE); // Which warp do you belong to?
    req_ind *= NUM_REQ_PER_LOOP;
    int w_tid = local_tid % WARP_SIZE;
    int masked_ind = w_tid % THREAD_PER_HDR_COPY;

    // Load packet response headers from shared to global memory
    for(unsigned i=0; i<NUM_REQ_PER_LOOP; ++i){
        res_ptr = (int *)TX_buf_ptrs[req_ind + i];
        pkt_hdr_ptr = (int *)(&mpi[req_ind + i].nmch);
        res_ptr[masked_ind] = pkt_hdr_ptr[masked_ind];
    }

}

__device__ void parse_pkt(unsigned long long first_RX_buffer_ptr,
                          int local_tid, int logical_tid,
                          int thread_type, mod_pkt_info *mpi){

    int *req_ptr = NULL;
    int *pkt_hdr_ptr = NULL;
    char *pkt_hdr = NULL;
    int ehs = sizeof(ether_header);
    int ips = sizeof(ip_header);
    ip_header *iph;
    udp_header *udp;
    u_int16_t check;

    int req_ind = (int)(local_tid / WARP_SIZE); // Which warp do you belong to?
    req_ind *= NUM_REQ_PER_LOOP;
    int w_tid = local_tid % WARP_SIZE;
    int masked_ind = w_tid % THREAD_PER_HDR_COPY;

    // Load packet headers from global to shared memory *coalesced accesses*
    for(unsigned i=0; i<NUM_REQ_PER_LOOP; ++i){
        req_ptr = (int *)( first_RX_buffer_ptr  +  ((req_ind + i)*RX_BUFFER_SZ) );

        pkt_hdr_ptr = (int *)(&mpi[req_ind + i].nmch);
        pkt_hdr_ptr[masked_ind] = req_ptr[masked_ind];
    }

    __syncthreads();

    // The packet header contents are all in shared memory, now verify the packet contents (still in global mem)
    mpi[logical_tid].is_get_req = 1; // Assume all are UDP Memcached GET requests
    if(thread_type == 0){
        pkt_hdr = (char *)&mpi[logical_tid].nmch;
        iph = (ip_header *)(pkt_hdr + ehs);
        udp = (udp_header *)(pkt_hdr + ehs + ips);

        // Check that the packet was for the GPU
        if(G_NTOHS(udp->dest) != UDP_PORT)
            mpi[logical_tid].is_get_req = 0;

        // Verify the checksum
        check = wrapsum(in_cksum((unsigned char *)iph, 4*(iph->version & 0x0F), 0));

        if(check != 0){
            mpi[logical_tid].is_get_req = 0;
        }
    }
}


__device__ void create_response_header(mod_pkt_info *mpi, int helper_tid){
    // m_res points to correct response memory for this helper_thread
    // mpi contains unmodified packet header, modify in shared memory

    // Elements to swap
    u_int8_t  ether_swap;
    u_int16_t ip_addr1;
    u_int16_t ip_addr2;
    u_int16_t udp_port;


    char *header = (char *)(&mpi->nmch);
    ether_header *eh = (ether_header *)header;
    ip_header *iph = (ip_header *)&header[14];
    udp_header *uh = (udp_header *)&header[34];

    // Swap ether
    for(unsigned i=0; i<ETH_ALEN; ++i){
        ether_swap = eh->ether_shost[i];
        eh->ether_shost[i] = eh->ether_dhost[i];
        eh->ether_dhost[i] = ether_swap;
    }

    // Swap IP
    ip_addr1 = iph->saddr1;
    ip_addr2 = iph->saddr2;
    iph->saddr1 = iph->daddr1;
    iph->saddr2 = iph->daddr2;
    iph->daddr1 = ip_addr1;
    iph->daddr2 = ip_addr2;

    iph->check = 0;

    // Swap UDP port
    udp_port = uh->source;
    uh->source = uh->dest;
    uh->dest = udp_port;
    uh->check = 0;

    iph->tot_len = G_HTONS((RESPONSE_SIZE - sizeof(ether_header)));// FIXME: Use info about item to generate this val in the next stage
    uh->len = G_HTONS((RESPONSE_SIZE - sizeof(ether_header) - sizeof(ip_header)));
    iph->check = wrapsum(in_cksum((unsigned char *)iph, 4*(iph->version & 0x0F), 0));

    return;

}


__device__ void populate_response(size_t *res_mem, mod_pkt_info *mpi,
                                  int tid, int helper_tid, int group_id,
                                  int *item_is_found, unsigned thread_type,
                                  int cta_id) {
    int *res_ptr = NULL;
    int *pkt_hdr_ptr = NULL;

    int req_ind = (int)(tid / WARP_SIZE); // Which warp does this thread belong to
    req_ind *= NUM_REQ_PER_LOOP;
    int local_tid = tid % WARP_SIZE;
    int masked_ind = local_tid % THREAD_PER_HDR_COPY;

    pkt_res_memc_hdr *start_response_pkt_hdr_mem = (pkt_res_memc_hdr *)(res_mem + NUM_REQUESTS_PER_BATCH);
    pkt_res_memc_hdr *response_pkt_hdr_mem = (pkt_res_memc_hdr *)&start_response_pkt_hdr_mem[cta_id*NUM_REQUESTS_PER_GROUP];

    // Store packet response headers from shared to global memory
    for(unsigned i=0; i<NUM_REQ_PER_LOOP; ++i){
        pkt_hdr_ptr = (int *)(&mpi[req_ind + i].nmch);
        res_ptr = (int *)&response_pkt_hdr_mem[req_ind + i];
        res_ptr[masked_ind] = pkt_hdr_ptr[masked_ind];
    }
}












































