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
 * gpu_km.h
 */

#include <linux/module.h>
#include <linux/vmalloc.h>
#include <linux/kernel.h>
#include <linux/socket.h>
#include <linux/skbuff.h>
#include <linux/rtnetlink.h>
#include <linux/in.h>
#include <linux/inet.h>
#include <linux/in6.h>
#include <linux/init.h>
#include <linux/filter.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/list.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/proc_fs.h>
#include <linux/if_arp.h>
#include <linux/if_vlan.h>
#include <net/xfrm.h>
#include <net/sock.h>
#include <asm/io.h>		
#ifdef CONFIG_INET
#include <net/inet_common.h>
#endif
#include <net/ip.h>
#include <net/ipv6.h>
#include <linux/pci.h>
#include <asm/shmparam.h>
#include <linux/notifier.h>
#include <linux/inetdevice.h>

#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <asm/uaccess.h>
#include <asm/errno.h>
#include <linux/pagemap.h>
#include <linux/moduleparam.h>
#include <linux/types.h>	/* size_t */
#include <linux/proc_fs.h>
#include <linux/fcntl.h>	/* O_ACCMODE */
#include <linux/seq_file.h>
#include <linux/cdev.h>
#include <linux/ktime.h>
#include <asm/atomic.h>
#include <linux/semaphore.h>
#include<linux/time.h>
#include <linux/delay.h>

#include "nv-p2p.h"

#include "gpu_km_shared.h"

#define GNOM_DEBUG // Enable for GNoM Debugging kernel print statements

#define GPU_KM_MAJOR 96
#define DEVICE_NAME "gpu_km"


/* Idea to repurpose an unused Linux network structure pointer borrowed from NTOP PF_RING */
#define hook_to_gpu_mod atalk_ptr

#define UBC_GPU	96

#define GNOM_RX 0
#define GNOM_TX 1

#ifndef VM_RESERVED
# define  VM_RESERVED   (VM_DONTEXPAND | VM_DONTDUMP)
#endif

#ifdef SINGLE_GPU_PKT_PTR
#define K_PAGE_ORDER 1      // Only need one page (8 byte/ptr)*(512 in-flight batches) = 8B per batch
#else
#define K_PAGE_ORDER 10     // 2x overallocation - Need one page per batch and 512 pages for 512 in-flight batches (8 b/ptr)*(512 req/batch)*(512 in-flight batches) = 4KB per batch
#endif

#define CPU_PAGE_SIZE 4096


/********************************************************************************/
/********************************************************************************/
// Functions
/********************************************************************************/
/********************************************************************************/
typedef int (*gpu_set_buffers_t)(void **gpu_buf, uint64_t *cuda_addr, void *dma, int *page_offset, int bid, int is_tx);
typedef int (*is_gpu_ready_t)(int *num_buf, uint64_t **buffer_pointers, bool (*recycle_grxb_batch)(int batch_ind, int num_req), int (*ixgbe_reg)(int cmd));
typedef int (*is_gpu_tx_ready_t)(int *num_buf, uint64_t **buffer_pointers, int (*ixgbe_gnom_tx)(int batch_ind, int buf_ind, unsigned int size));
typedef int (*gpu_dispatch_work_t)(int rx_b_ind, int tx_b_ind, int num_req);

typedef int (*driver_get_dma_info)(struct page **kernel_page, void **kernel_va_addr, int set_finished);
typedef int (*set_device_t)(struct device *dev);

// IXGBE RPC functions
//static bool (* ixgbe_callback_recycle_grxb_batch)(int batch_ind, int num_req);  // Recycle batch of GRX buffers
//static int (* ixgbe_callback_register)(int cmd);                                // Used to issue commands to the IXGBE driver
//static int (* gnom_tx)(int batch_ind, int buf_ind, unsigned int size);          // TX Send function
/********************************************************************************/
/********************************************************************************/

/********************************************************************************/
/********************************************************************************/
// Data structures
/********************************************************************************/
/********************************************************************************/
struct mmap_info {
    char *data;
    char *data2;
    int reference_cnt;
};

typedef struct _ubc_gpu_hooks_ {
	unsigned magic;
	gpu_set_buffers_t gpu_set_buffers;
	is_gpu_ready_t is_gpu_rx_ready;
	is_gpu_tx_ready_t is_gpu_tx_ready;
	gpu_dispatch_work_t gpu_dispatch_work;

	driver_get_dma_info get_dma_info;
    set_device_t gpu_set_device;
    
}ubc_gpu_hooks;

typedef struct _gpu_buf_ {
	uint64_t cuda_addr;
	void *host_addr;
	dma_addr_t dma;
	unsigned page_offset;
    struct page *user_pg;
}gpu_buf_t;

typedef struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
	uint64_t p2pToken;
	uint32_t vaSpaceToken;
} CUDA_POINTER_ATTRIBUTE_P2P_TOKENS;


typedef struct _network_requests_{
	int num_reqs; // Number of requests in buffer
	void *cuda_buffers[MAX_REQ_BATCH_SIZE];	// Pointers to CUDA buffers (returned by CUDA malloc)
	int cuda_buffer_ids[MAX_REQ_BATCH_SIZE];
}network_requests;

typedef struct _lw_network_requests_{
	int num_reqs;
	int batch_id;
	int rx_batch_ind;
	int tx_batch_ind;
	ktime_t dispatch_time;
}lw_network_requests;


typedef struct _kernel_args_ {
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS m_tokens;
    uint64_t m_addr;	// Virtual GPU address of the mapped GPU page
    uint64_t m_size;	// Size of buffer within page
}kernel_args;

typedef struct _cpu_kernel_args_ {
    void *user_page_va;
    uint64_t cuda_page_va;
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
    
    int buffer_type; // 0 = RX, 1 = TX
}kernel_args_hdr;
/********************************************************************************/
/********************************************************************************/


/**********************************/
/*********** GNoM Debug ***********/
/**********************************/
#ifdef GNOM_DEBUG

void print_pkt_hdr(void *data, int is_tcp);
void print_skb_info(struct sk_buff *skb);

typedef struct _ether_header{
  u_int8_t  ether_dhost[ETH_ALEN];  /* destination eth addr */
  u_int8_t  ether_shost[ETH_ALEN];  /* source ether addr    */
  u_int16_t ether_type;             /* packet type ID field */
}ether_header /*__attribute__((packed))*/;

typedef struct _ip_header {
  u_int8_t  version;            /* version */           // Version+ihl = 8 bits, so replace ihl with 8bit version
  //u_int32_t ihl:4;            /* header length */

  u_int8_t  tos;            /* type of service */
  u_int16_t tot_len;            /* total length */
  u_int16_t id;         /* identification */
  u_int16_t frag_off;           /* fragment offset field */
  u_int8_t  ttl;            /* time to live */
  u_int8_t  protocol;           /* protocol */
  u_int16_t check;          /* checksum */

  u_int32_t saddr;
  u_int32_t daddr;  /* source and dest address */
}ip_header;

typedef struct _udp_header {
  u_int16_t source;     /* source port */
  u_int16_t dest;       /* destination port */
  u_int16_t len;        /* udp length */
  u_int16_t check;      /* udp checksum */
}udp_header;
#endif



