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
 * gnom_ixgbe.h
 */

#ifndef _GNOM_IXGBE_H_
#define _GNOM_IXGBE_H_


#include "gpu_km.h"

/***************************/
/********* DEFINES *********/
/***************************/

//////////////////////////////////////////////////////
// These need to be set to the PCI device name for 
// the NIC ports. E.g., the NIC in our system has 
// ports mapped to 0000:04:00.0 and 0000:04:00.1.
// "eth2" = 0000:04:00.0 is used for RX
// Experimental: "eth3" = 0000:04:00.1 is used for TX (when DO_GNOM_TX set)

// TODO: Remove hardcoding
#define GNOM_RX_DEVICE_PCI_NAME "0000:04:00.0"
#define GNOM_TX_DEVICE_PCI_NAME "0000:04:00.1"

#define GNOM_RX_NAME "eth2"
#define GNOM_TX_NAME "eth3"

/////////////////////////////////////////////////////


#define NUM_RX_QUEUE_BUF 512

/*****************************/
/********* Structures ********/
/*****************************/

typedef enum {GNOM_FREE = 0, GNOM_ON_NIC, GNOM_ON_GPU} gnom_buf_state;

typedef struct _nic_gpu_ring_buf_{
    unsigned bid;               // Ring ID
    gnom_buf_state b_state;		// State of GNoM Buffer
    void *m_page_addr;          // Mapped Kernel Virtual Page Address
    dma_addr_t dma;             // Bus address of GPU buffer
    uint64_t m_gpu_host_addr;   // Address returned by CUDA malloc
    unsigned page_offset;       // Offset of buffer in page
}nic_gpu_ring_buf;

typedef struct _gnom_ixgbe_stats_{
    long long rx_num_req_dispatched;
    long long rx_num_buf_recycled;
    long long rx_num_buf_alloc;
    long long rx_failed_map_count;

    long long tx_num_buf_recycled;
    long long tx_num_buf_fetched;
    long long tx_failed_map_count;
}gnom_ixgbe_stats;

typedef struct _gnom_ring_config_{
    int is_set;
    int num_gpu_buffers;
    int num_gpu_rings;
    int num_gpu_buf_per_ring;

    int num_gpu_bins;
    int num_gpu_bufs_last_bin;
}gnom_ring_config;

typedef struct _gnom_ring_{
    nic_gpu_ring_buf *g_ring_bufs;
    nic_gpu_ring_buf **g_ring_free_bufs;
    int free_head;
    int free_tail;
}gnom_ring;

typedef struct _new_gnom_ring_{
    nic_gpu_ring_buf **g_ring_bufs;
    nic_gpu_ring_buf ***g_ring_free_bufs;

    int free_head_bin;
    int free_head_buf;
    int free_tail_bin;
    int free_tail_buf;

}new_gnom_ring;

typedef struct _gnom_req_batch_{
    uint64_t    *bufs;              // IF SINGLE_GPU_PKT_PTR then points to first buffer pointer, else points to an array with buffer pointers for complete batch
    int         buf_ids[NUM_RX_QUEUE_BUF][MAX_REQ_BATCH_SIZE];
    int         req_batch_ind;
    int         req_buf_ind;
}gnom_req_batch;

/*******************************************************/
/**************** Function Declarations ****************/
/*******************************************************/
extern int gnom_manually_add_fdir_entry(struct ixgbe_adapter *adapter,
        struct ethtool_rxnfc *cmd);
int ixgbe_set_mac(struct net_device *netdev, void *p);

int setup_gpu_rx_env(struct ixgbe_adapter *adapter);
int setup_gpu_tx_env(struct ixgbe_adapter *adapter);

int ixgbe_gpu_km_callback(int cmd);
bool ixgbe_recycle_grxb_batch(int batch_ind, int num_req);
bool ixgbe_recycle_gtxb(int batch_id, int buf_id);
int ixgbe_gnom_tx_send(int batch_ind, int buf_ind, unsigned int size);

int set_gnom_rx_adapter(struct ixgbe_adapter *adapter);
int set_gnom_tx_adapter(struct ixgbe_adapter *adapter);

void init_tx_lock(void);
void gnom_tx_lock(void);
void gnom_tx_unlock(void);


inline void gnom_inc_rx_alloc(int num);
inline void gnom_inc_rx_recycled(int num);
inline void gnom_inc_rx_map_failed(int num);
inline void gnom_inc_req_dispatched(int num);
inline void gnom_inc_num_tx_fetched(int num);
inline void gnom_inc_num_tx_recycled(int num);
inline void gnom_inc_num_tx_failed_map(int num);

int is_gnom_rx_enabled(void);
int is_gnom_tx_enabled(void);

void gnom_cleanup_rx(void);
void gnom_cleanup_tx(void);

bool ixgbe_alloc_gpu_mapped_page(struct ixgbe_ring *rx_ring, struct ixgbe_rx_buffer *bi);
int ixgbe_clean_gpu_rx_irq(struct ixgbe_q_vector *q_vector,
                            struct ixgbe_ring *rx_ring,
                            int budget);

int verify_pkt(void *data);
void gnom_print_stats(void);

int test_tx(void *data, dma_addr_t dma);

/*************************************************************************/
/* Custom GNoM Functions for extracting packet header fields without SKB */
/*************************************************************************/
static inline struct iphdr *gnom_ip_hdr(const void *pkt){
    return (struct iphdr *)(pkt + sizeof(struct ethhdr));
}

static inline uint8_t gnom_ip_hdr_len(const void *pkt){
    uint8_t len;
    struct iphdr *iph = (struct iphdr *)(pkt + sizeof(struct ethhdr));
    len = iph->ihl * sizeof(int); // Number of 32-bit words in IP header

    return len;
}

static inline struct tcphdr *gnom_tcp_hdr(const void *pkt){
    return (struct tcphdr *)(pkt + sizeof(struct ethhdr) + gnom_ip_hdr_len(pkt));
}

static inline unsigned int gnom_tcp_hdrlen(const void *pkt){
    return gnom_tcp_hdr(pkt)->doff * 4;
}

static inline unsigned short gnom_eth_proto(const void *pkt){
    return ((struct ethhdr *)pkt)->h_proto;
}



#endif
