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
 * gnom_ixgbe.c
 */

#include "ixgbe.h"
#include "ixgbe_type.h"
#include "gnom_ixgbe.h"


#include <linux/types.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/netdevice.h>
#include <linux/vmalloc.h>
#include <linux/highmem.h>
#include <linux/string.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/pkt_sched.h>
#include <linux/ipv6.h>
#ifdef NETIF_F_TSO
#include <net/checksum.h>
#ifdef NETIF_F_TSO6
#include <net/ip6_checksum.h>
#endif
#endif
#include <linux/if_bridge.h>
#include <linux/semaphore.h>
#include <linux/ktime.h>
#include <linux/netdevice.h>
#include <linux/udp.h>
#include <linux/if_ether.h>

// Taken from ixgbe_main.c
#define IXGBE_TXD_CMD (IXGBE_TXD_CMD_EOP | \
               IXGBE_TXD_CMD_RS)


/******************************************************/
/****************** GLOBAL VARIABLES ******************/
/******************************************************/
// GNoM Ring Buffer structures
gnom_ring grx_ring;                 // RX
gnom_ring gtx_ring;                 // TX

// GNoM Ring configurations
gnom_ring_config grx_ring_config;   // RX
gnom_ring_config gtx_ring_config;   // TX

// GRX request batch info
gnom_req_batch grxb;                // RX
gnom_req_batch gtxb;                // TX

// Adapter assigned to GNoM
struct ixgbe_adapter *gnom_rx_adapter = NULL;
struct ixgbe_adapter *gnom_tx_adapter = NULL;

struct ixgbe_ring *gnom_tx_ring;

// GNoM Stats structure
gnom_ixgbe_stats gnom_stats = {.rx_num_buf_alloc = 0,
                               .rx_num_buf_recycled = 0,
                               .rx_num_req_dispatched = 0,
                               .rx_failed_map_count = 0};


struct semaphore gnom_global_tx_lock;


// Testing
static dma_addr_t gnom_dma_addr = 0;
void *gnom_kv_addr = NULL;
struct page *gnom_page = NULL;

/******************************************************/


unsigned long long tx_total_time = 0;
unsigned long long tx_total_count = 0;

unsigned long long rx_total_time = 0;
unsigned long long rx_total_count = 0;


//static unsigned long long sanity_val = 0;
static unsigned long long sanity_sum = 0;

/******************************************************/
/*********************** STATS ************************/
/******************************************************/

inline void gnom_inc_rx_alloc(int num){
    gnom_stats.rx_num_buf_alloc+=num;
}

inline void gnom_inc_rx_recycled(int num){
    gnom_stats.rx_num_buf_recycled+=num;
}

inline void gnom_inc_rx_map_failed(int num){
    gnom_stats.rx_failed_map_count+=num;
//    if(gnom_stats.rx_failed_map_count % 512 == 0){
//      printk("[GNoM_ixgbe]: Failed to alloc_gpu_mapped_page (%u) ( dispatched=%lld, recycled=%lld, alloced=%lld, diff1=%lld, diff2=%lld)...\n", gnom_stats.rx_failed_map_count,
//              gnom_stats.rx_gnom_ixgbe_stats, gnom_stats.rx_num_buf_recycled, gnom_stats.rx_num_buf_alloc, gnom_stats.rx_num_req_dispatched - gnom_stats.rx_num_buf_recycled, gnom_stats.rx_num_buf_alloc-gnom_stats.rx_num_buf_recycled );
//    }
}

inline void gnom_inc_req_dispatched(int num){
    gnom_stats.rx_num_req_dispatched+=num;
}

inline void gnom_inc_num_tx_fetched(int num){
    gnom_stats.tx_num_buf_fetched += num;
}
inline void gnom_inc_num_tx_recycled(int num){
    gnom_stats.tx_num_buf_recycled += num;
}
inline void gnom_inc_num_tx_failed_map(int num){
    gnom_stats.tx_failed_map_count += num;
}

void gnom_print_stats(){
    int i=0;
    unsigned long long sum = 0;

    printk("[GNOM_ixgbe]: GNoM Stats:\n");
    printk("[GNOM_ixgbe]: RX:\n");
    printk("\tNum RX alloced: %lld\n\tNum RX recycled: %lld\n\tNum RX dispatch: %lld\n\tNum RX failed map: %lld\n",
            gnom_stats.rx_num_buf_alloc,
            gnom_stats.rx_num_buf_recycled,
            gnom_stats.rx_num_req_dispatched,
            gnom_stats.rx_failed_map_count);
    printk("[GNOM_ixgbe]: TX:\n");
    printk("\tNum TX fetched: %lld\n\tNum TX recycled: %lld\n\tNum TX failed map: %lld\n",
            gnom_stats.tx_num_buf_fetched,
            gnom_stats.tx_num_buf_recycled,
            gnom_stats.tx_failed_map_count);


    if(rx_total_count > 0){
        printk("[GNOM_ixgbe]: RX: Total time for %lld buffers: %lld ns\n", rx_total_time, rx_total_count);
    }

    if(tx_total_count > 0){
        printk("[GNOM_ixgbe]: TX: Total time for %lld buffers: %lld ns\n", tx_total_time, tx_total_count);

        for(i=0; i<tx_total_count; ++i){
            sum += i;
        }

        printk("[GNOM_ixgbe]: TX IRQ expected sanity_sum: %llu, actual %llu\n", sum, sanity_sum);

    }

}

/******************************************************/
/*********************** GLOBAL ***********************/
/******************************************************/

int set_gnom_rx_adapter(struct ixgbe_adapter *adapter){
    if(!gnom_rx_adapter){
        gnom_rx_adapter = adapter;
        return 0;
    }else{
        return -1;
    }
}

int set_gnom_tx_adapter(struct ixgbe_adapter *adapter){
    if(!gnom_tx_adapter){
        gnom_tx_adapter = adapter;
        return 0;
    }else{
        return -1;
    }
}

void init_tx_lock(){
    sema_init(&gnom_global_tx_lock, 1);
}

void gnom_tx_lock(){
    down(&gnom_global_tx_lock);
}

void gnom_tx_unlock(){
    up(&gnom_global_tx_lock);
}

int is_gnom_rx_enabled(){
    return grx_ring_config.is_set;
}

int is_gnom_tx_enabled(){
    return gtx_ring_config.is_set;
}

void gnom_cleanup_rx(){
    if(grx_ring.g_ring_bufs)
        vfree(grx_ring.g_ring_bufs);
    if(grx_ring.g_ring_free_bufs)
        vfree(grx_ring.g_ring_free_bufs);
}

void gnom_cleanup_tx(){
    if(gtx_ring.g_ring_bufs)
        vfree(gtx_ring.g_ring_bufs);
    if(gtx_ring.g_ring_free_bufs)
        vfree(gtx_ring.g_ring_free_bufs);
}

int ixgbe_gpu_km_callback(int cmd){

    struct net_device *netdev = gnom_tx_adapter->netdev;

    struct ixgbe_ring *ring;
    ubc_gpu_hooks *hook;
    struct device *dev = pci_dev_to_dev(gnom_tx_adapter->pdev);

    hook = netdev->hook_to_gpu_mod;

    switch(cmd){
    case 0: // Reconfigure the GNoM Adapter. ixgbe_up() calls hooks back to GNoM-KM to enable/disable GNoM capture mode.
        if(!gnom_rx_adapter || ! gnom_tx_adapter)
            goto adapter_not_set_err;

        printk("[GNoM_ixgbe]: Restarting NIC from GPU_KM\n");

        ixgbe_down(gnom_rx_adapter);
#ifdef DO_GNOM_TX
        ixgbe_down(gnom_tx_adapter);
#endif

        gnom_cleanup_rx();
#ifdef DO_GNOM_TX
        gnom_cleanup_tx();
#endif
        msleep(10);

        ixgbe_up(gnom_rx_adapter);
#ifdef DO_GNOM_TX
        ixgbe_up(gnom_tx_adapter);
#endif
        break;

    case 1: // DEBUG TEST MAP CPU BUFFER
        ring = gnom_tx_adapter->tx_ring[0];

        if(!hook || !gnom_tx_adapter){
            printk("[GNoM_ixgbe]: ERROR: hook or adapter not set...\n");
            return -1;
        }

        gnom_tx_lock();
        hook->get_dma_info(&gnom_page, &gnom_kv_addr, 0);

        printk("[GNoM_ixgbe]: gnom_page: %p, gnom_addr: %p (dev=%p, adapter_dev=%p)\n", gnom_page, gnom_kv_addr, ring->dev, pci_dev_to_dev(gnom_tx_adapter->pdev));


        gnom_dma_addr = dma_map_page(dev, gnom_page, 0, 1024, DMA_TO_DEVICE);

        if (dma_mapping_error(dev, gnom_dma_addr)){
            gnom_tx_unlock();
            return -1;
        }

        // Make sure everything is synced for the CPU
        dma_sync_single_range_for_cpu(dev,
                          gnom_dma_addr,
                          0,
                          1024,
                          DMA_TO_DEVICE);

        gnom_tx_unlock();
        break;

    case 2: // DEBUG TEST UNMAP CPU BUFFER

        if(gnom_page == NULL){
            printk("[GNoM_ixgbe]: ERROR in unmap: gnom_dma_addr not set yet...\n");
            return -1;
        }

        gnom_tx_lock();
        ring = gnom_tx_adapter->tx_ring[0];

        dma_unmap_page(dev, gnom_dma_addr,
                 1024,
                 DMA_TO_DEVICE);
        gnom_tx_unlock();

        break;


    case 3: // DEBUG TEST SEND FROM GPU MAPPED CPU BUFFER
        gnom_tx_lock();
        if(gnom_kv_addr != NULL){
            printk("[GNoM_ixgbe]: Trying send...\n");
            test_tx(gnom_kv_addr, gnom_dma_addr);
        }
        gnom_tx_unlock();

        break;

    case 4: // Print GNoM_ixgbe statistics
        gnom_print_stats();
        break;
    default:
        break;
    }

    return 0;


adapter_not_set_err:
    printk("[GNoM_ixgbe]: IXGBE Callback registration failed - gnom_rx_adapter (%p) or gnom_tx_adapter (%p) not set\n", gnom_rx_adapter, gnom_tx_adapter );

    return -1;
}

static void ixgbe_gnom_tx_csum(struct ixgbe_ring *tx_ring,
              struct ixgbe_tx_buffer *first,
              void *data){
    u32 vlan_macip_lens = 0;
    u32 mss_l4len_idx = 0;
    u32 type_tucmd = 0;
    u8 l4_hdr = 0;


    switch (first->protocol) {
    case __constant_htons(ETH_P_IP):
        //vlan_macip_lens |= skb_network_header_len(skb);
        vlan_macip_lens |= gnom_ip_hdr_len(data);
        type_tucmd |= IXGBE_ADVTXD_TUCMD_IPV4;
        //l4_hdr = ip_hdr(skb)->protocol;
        l4_hdr = gnom_ip_hdr(data)->protocol;
        break;

    default:
        printk("[INTEL_IXGBE]: Error on ixgbe_gnom_tx_csum - Not ETH_P_IP packet in GNoM Flow...\n");
        if (unlikely(net_ratelimit())) {
            dev_warn(tx_ring->dev,
             "partial checksum but proto=%x!\n",
             first->protocol);
        }
        break;
    }

    switch (l4_hdr) {
    case IPPROTO_TCP:
        type_tucmd |= IXGBE_ADVTXD_TUCMD_L4T_TCP;
        //mss_l4len_idx = tcp_hdrlen(skb) << IXGBE_ADVTXD_L4LEN_SHIFT;
        mss_l4len_idx = gnom_tcp_hdrlen(data) << IXGBE_ADVTXD_L4LEN_SHIFT;
        break;
    case IPPROTO_UDP:
        mss_l4len_idx = sizeof(struct udphdr) << IXGBE_ADVTXD_L4LEN_SHIFT;
        break;
    default:
        printk("[INTEL_IXGBE]: Error on ixgbe_gnom_tx_csum\n");
        if (unlikely(net_ratelimit())) {
            dev_warn(tx_ring->dev,
             "partial checksum but l4 proto=%x!\n",
             l4_hdr);
        }
        break;
    }

    /* update TX checksum flag */
    first->tx_flags |= IXGBE_TX_FLAGS_CSUM;

    /* vlan_macip_lens: MACLEN, VLAN tag */
    //vlan_macip_lens |= skb_network_offset(skb) << IXGBE_ADVTXD_MACLEN_SHIFT;
    // Need to figure out if skb_network_offset(skb) ever changes. Seems like it's always set to sizeof(ethhdr) == 14
    vlan_macip_lens |= sizeof(struct ethhdr) << IXGBE_ADVTXD_MACLEN_SHIFT;
    vlan_macip_lens |= first->tx_flags & IXGBE_TX_FLAGS_VLAN_MASK;

    ixgbe_tx_ctxtdesc(tx_ring, vlan_macip_lens, 0, type_tucmd, mss_l4len_idx);

}

/******************************************************/
/************************* RX *************************/
/******************************************************/

int setup_gpu_rx_env(struct ixgbe_adapter *adapter){
    struct net_device *netdev = adapter->netdev;
    struct ixgbe_ring *rx_ring;
    ubc_gpu_hooks *hook;
    struct ethtool_rxnfc cmd;
    unsigned i = 0;
    int ret = 0;

    hook = netdev->hook_to_gpu_mod;
    if(!hook || !hook->is_gpu_rx_ready(&grx_ring_config.num_gpu_buffers, &grxb.bufs,
                &ixgbe_recycle_grxb_batch, &ixgbe_gpu_km_callback) ){
        if(!hook) printk("[GNoM_ixgbe]: Failed Hook...(hook = %p)\n", hook);
        else printk("[GNoM_ixgbe]: GPU RX buffers not set yet! (hook = %p)\n", hook);

        goto err;
    }

    if(grx_ring_config.num_gpu_buffers <= 0){
        printk("[GNoM_ixgbe]: num gpu buffers wrong...\n");
        goto err;
    }

    printk("[GNoM_ixgbe]: GRXB.bufs = %p\n", grxb.bufs);

    // Can configure GNoM to use multiple RX rings, or to leave rings for the standard CPU path. SoCC paper used a single ring total (for GPU). 
    // TODO: There have been multiple changes since this was tested, need to verify if this still works. 
    grx_ring_config.num_gpu_rings = 1;

    grxb.req_buf_ind = 0;
    grxb.req_batch_ind = 0;

    printk("[GNoM_ixgbe]: GPU_km is up and running with %d GPU buffers for us\n", grx_ring_config.num_gpu_buffers);

    /*********************/
    
    // GPU module is up and running, populate GPU buffers, initialize structures
    grx_ring.g_ring_bufs = (nic_gpu_ring_buf *)vmalloc((grx_ring_config.num_gpu_buffers+1)*sizeof(nic_gpu_ring_buf)); // Storage for buffers
    grx_ring.g_ring_free_bufs = (nic_gpu_ring_buf **)vmalloc((grx_ring_config.num_gpu_buffers+1)*sizeof(nic_gpu_ring_buf *)); // List of free buffers

    if(!grx_ring.g_ring_bufs || !grx_ring.g_ring_free_bufs)
        goto gpu_alloc_err;

    grx_ring.free_head = grx_ring_config.num_gpu_buffers; // Next index to push to
    grx_ring.free_tail = 0; // Next index to pull from


    for(i=0; i<grx_ring_config.num_gpu_buffers; ++i){
        grx_ring.g_ring_bufs[i].bid = i;
        grx_ring.g_ring_bufs[i].b_state = GNOM_FREE;
        if(hook->gpu_set_buffers(&grx_ring.g_ring_bufs[i].m_page_addr,          // Virtual CPU mmaped page address
                                    &grx_ring.g_ring_bufs[i].m_gpu_host_addr,   // CUDA virtual address
                                    &grx_ring.g_ring_bufs[i].dma,               // CUDA physical address
                                    &grx_ring.g_ring_bufs[i].page_offset,       // Page offset
                                    i,                                          // BID
                                    GNOM_RX))                                   // RX buffer
            goto gpu_alloc_err;

        grx_ring.g_ring_free_bufs[i] = &grx_ring.g_ring_bufs[i]; // Initially fill free list with all buffers
    }


    /*********************/

    if(adapter->num_rx_queues > 1){
        // GNoM can be configured to use packet filters to force GNoM packets (based on the UDP receive port). 
        printk("[GNoM_ixgbe]: Detected multiple RX queues. Need to verify the multi-RX queue config still works\n");
        printk("[GNoM_ixgbe]: Setting up hardware packet filters.\n");

        // Initialize Filters - avoiding going through ethtool to manually install packet filters
        memset(&cmd, 0, sizeof(struct ethtool_rxnfc)); // Clear everything
        cmd.cmd = 50;
        cmd.flow_type = 32767;
        cmd.rule_cnt = 0;
        i=0;
        
        if(grx_ring_config.num_gpu_rings > adapter->num_rx_queues){
            printk("[GNoM_ixgbe]: Error - Trying to have more GNoM RX rings than the NIC RX queues (%d > %d)\n", grx_ring_config.num_gpu_rings, adapter->num_rx_queues);
            goto gpu_filter_err;
        }

        for(i=0; i<grx_ring_config.num_gpu_rings; ++i){ // Set GNoM GPU rings from back to front
            rx_ring = adapter->rx_ring[adapter->num_rx_queues - i - 1];

            printk("\[GNoM_ixgbe]:\tSetting rx_ring (%d) to gpu ring\n", adapter->num_rx_queues - i - 1);
            rx_ring->is_gpu_ring = 1; // Set this ring as a GPU ring

            // Set a rule for this ring
            cmd.fs.flow_type = 2; // UDP request
            cmd.fs.location = (2045 - i); // SW location # (starts at 2045, assuming no other filters to start)

            cmd.fs.ring_cookie = adapter->num_rx_queues - i - 1; // First NUM_GPU_RINGS set to GPU rings

            cmd.fs.h_u.udp_ip4_spec.pdst = htons(9960 + i); // Move from port 9600->9600+(NUM_GPU_RINGS-1)
            cmd.fs.m_u.udp_ip4_spec.pdst = 0xFFFF;

            ret = gnom_manually_add_fdir_entry(adapter, &cmd);
            if(ret){
                printk("[GNoM_ixgbe]: Error adding hardware packet filter...(%d)\n", ret);
                goto gpu_filter_err;
            }
        }
    }else{
        rx_ring = adapter->rx_ring[0];
        rx_ring->is_gpu_ring = 1;
        printk("[GNoM_ixgbe]: Single NIC RX queue in use for GNoM GPU RX. \n");
    }

    rmb(); // Ensure all buffers are initialized before registering them with the NIC

    // Set the enable flag for GRX
    grx_ring_config.is_set = 1;

    printk("[GNoM_ixgbe]: \t%d GPU RX buffers successfully initialized in NIC driver on %s\n", grx_ring_config.num_gpu_buffers, netdev->name);

    return 0;

gpu_alloc_err:
    printk("[GNoM_ixgbe]: Error: Can't allocate the %d'th GPU buffer...\n", i);
gpu_filter_err:
    gnom_cleanup_rx();

err:
    for(i=0; i<adapter->num_rx_queues; ++i){
        rx_ring = adapter->rx_ring[i];
        rx_ring->is_gpu_ring = 0; // Set this ring as a CPU ring again
    }

    return 1;
}

bool ixgbe_recycle_grxb_batch(int batch_ind, int num_req){
    // No need to check errors here since we have enough space to hold all free buffers.
    
    unsigned i;
    int bid;
    
    for(i=0; i<num_req; ++i){
        bid = grxb.buf_ids[batch_ind][i];
        if(unlikely(bid == -1)) break;

        grx_ring.g_ring_free_bufs[grx_ring.free_head] = &grx_ring.g_ring_bufs[bid];
        grx_ring.g_ring_free_bufs[grx_ring.free_head]->b_state = GNOM_FREE;

        wmb();

        // Update grx_ring.free_head (grx_ring_config.num_gpu_buffers+1 locations)
        grx_ring.free_head = ((grx_ring.free_head+1) <= grx_ring_config.num_gpu_buffers) ? (grx_ring.free_head + 1) : 0;

    }
    gnom_inc_rx_recycled(num_req);

    return true;
}

// Pull new buffer off of the free gpu buffer list, pop from head.
// Sync the GRXB with the NIC for RX. 
bool ixgbe_alloc_gpu_mapped_page(struct ixgbe_ring *rx_ring, struct ixgbe_rx_buffer *bi){
    nic_gpu_ring_buf* ring_buf;

    if(unlikely(grx_ring.free_tail == grx_ring.free_head )){ // No free buffers left :(. This leads to packet drops.
        return false;
    }

    ring_buf = grx_ring.g_ring_free_bufs[grx_ring.free_tail];   // Set buffer
    grx_ring.g_ring_free_bufs[grx_ring.free_tail] = NULL;

    // Update grx_ring.free_tail (grx_ring_config.num_gpu_buffers+1 locations)
    grx_ring.free_tail = ((grx_ring.free_tail+1) <= grx_ring_config.num_gpu_buffers) ? (grx_ring.free_tail + 1) : 0;

    ring_buf->b_state = GNOM_ON_NIC;

    // Update buffer_info to use GPU buffer
    //bi->page = virt_to_page(ring_buf->m_page_addr);
    bi->dma = ring_buf->dma;
    bi->page_offset = ring_buf->page_offset; // Need to remember our offset into the GPU page for this buffer (32 buf/pg)
    bi->gpu_buffer_id = ring_buf->bid; // So we can add it back to the freelist.
    bi->gpu_buffer_host_addr = (ring_buf->m_gpu_host_addr + ring_buf->page_offset);

    // Tesing the time the GRXBs are registered with the NIC
    //bi->pushed_to_nic_time = ktime_get();

    dma_sync_single_range_for_device(rx_ring->dev,
                                    bi->dma,
                                    bi->page_offset,
                                    /*ixgbe_rx_bufsz(rx_ring),*/
                                    RX_BUFFER_SZ,
                                    DMA_FROM_DEVICE);

    return true;
}

// Grab the most recent GRXB from the NIC with a valid RX packet. 
int ixgbe_fetch_gpu_rx_buffer(struct ixgbe_ring *rx_ring, union ixgbe_adv_rx_desc *rx_desc){
    struct ixgbe_rx_buffer *rx_buffer;

    rx_buffer = &rx_ring->rx_buffer_info[rx_ring->next_to_clean];


    // Likely need to do something with EOP eventually. Currently ignore.
    //if (likely(ixgbe_test_staterr(rx_desc, IXGBE_RXD_STAT_EOP)))
    //    goto dma_sync;

    // Sync buffer for GPU use
    dma_sync_single_range_for_cpu(rx_ring->dev,
                      rx_buffer->dma,
                      rx_buffer->page_offset, // Now pulling in the correct offset buffer
                      /*ixgbe_rx_bufsz(rx_ring),*/
                      RX_BUFFER_SZ,
                      DMA_FROM_DEVICE);

    return 0;
}


/**
 * ixgbe_clean_gpu_rx_irq - Clean completed descriptors from GPU Rx ring - bounce buf
 * @q_vector: structure containing interrupt and ring information
 * @rx_ring: rx descriptor ring to transact packets on
 * @budget: Total limit on number of packets to process
 *
 * This function provides a "bounce buffer" approach to Rx interrupt
 * processing.  The advantage to this is that on systems that have
 * expensive overhead for IOMMU access this provides a means of avoiding
 * it by maintaining the mapping of the page to the system.
 *
 * Returns amount of work completed.
 **/

//static int first = 1;
unsigned cnt = 0;
int last = 0;
ktime_t start, end;
unsigned long long actual_time = 0;
unsigned global_packet_count = 0;

// Function structure similar to ixgbe_clean_rx_irq in ixgbe_main.c
int ixgbe_clean_gpu_rx_irq(struct ixgbe_q_vector *q_vector,
                            struct ixgbe_ring *rx_ring,
                            int budget){

    unsigned int total_rx_packets = 0;
    u16 cleaned_count = ixgbe_desc_unused(rx_ring);
    union ixgbe_adv_rx_desc *rx_desc;
    struct ixgbe_rx_buffer *rx_buffer;
    int gnom_tx_batch_ind = -1;

    ubc_gpu_hooks *hook;
    struct net_device *netdev = rx_ring->netdev;
    hook = (ubc_gpu_hooks*)netdev->hook_to_gpu_mod;

    do{
        if (cleaned_count >= IXGBE_RX_BUFFER_WRITE) {
            ixgbe_alloc_rx_buffers(rx_ring, cleaned_count);
            cleaned_count = 0;
        }

        rx_desc = IXGBE_RX_DESC(rx_ring, rx_ring->next_to_clean);
        if (!ixgbe_test_staterr(rx_desc, IXGBE_RXD_STAT_DD))
            break;

        /*
         * This memory barrier is needed to keep us from reading
         * any other fields out of the rx_desc until we know the
         * RXD_STAT_DD bit is set
         */
        rmb();

        /* retrieve a buffer from the ring */
        /* Calling a new function which syncs the GPU buffer, but doesn't consider Linux SKBs*/
        ixgbe_fetch_gpu_rx_buffer(rx_ring, rx_desc);

        /* update budget accounting */
        total_rx_packets++;
        global_packet_count++;

        /* Check if we have enough requests to launch a CUDA kernel */
        rx_buffer = &rx_ring->rx_buffer_info[rx_ring->next_to_clean];

        // For the first buffer in the batch, save the initial buffer pointer
        if(unlikely(grxb.req_buf_ind == 0)){
            grxb.bufs[grxb.req_batch_ind] = rx_buffer->gpu_buffer_host_addr;
        }

        // Save the buffer ID for recycling
        grxb.buf_ids[grxb.req_batch_ind][grxb.req_buf_ind] = rx_buffer->gpu_buffer_id;

        // Move to the next buffer index for this batch
        grxb.req_buf_ind++;

        /*
        now = ktime_get();
        rx_total_time += ktime_to_ns(ktime_sub(now, rx_buffer->pushed_to_nic_time));
        */
        rx_total_count++;

        // TODO: Add a timer to ensure that packets don't remain batched for too long if the receive rate is low. 
        if(grxb.req_buf_ind >= MAX_REQ_BATCH_SIZE){ // Received enough packets, now lets do some work on the GPU
            if(unlikely(!hook)){
                printk("[GNoM_ixgbe]: Error: hook is Null in ixgbe_clean_gpu_rx_irq!\n");
            }else{
#ifdef DO_GNOM_TX
                // Fetch a batch of TX buffers for sending response messages (# of requests in batch = grxb.req_buf_ind).
                // TODO: Remove from RX path and into the TX path. Currently here for testing initial GPU TX. 
                gnom_tx_batch_ind = gnom_fetch_tx_buffer_batch(grxb.req_buf_ind);

                if(unlikely(gnom_tx_batch_ind < 0)){
                    printk("[GNoM_ixgbe]: Error gnom_tx_batch_ind  == %d...\n", gnom_tx_batch_ind);
                    gnom_tx_batch_ind = 0;
                }
#endif

                // Pass pkt buffers/# pkts to GPU_kmi
                if(unlikely(hook->gpu_dispatch_work(grxb.req_batch_ind, gnom_tx_batch_ind, grxb.req_buf_ind)))
                    printk("[GNoM_ixgbe]: Error: Couldn't dispatch work to GPU_km...\n");

                gnom_inc_req_dispatched(grxb.req_buf_ind);

                // Move to next batch
                grxb.req_batch_ind = (grxb.req_batch_ind + 1) % NUM_RX_QUEUE_BUF;

                // Reset buffer for next batch
                grxb.req_buf_ind = 0;

            }
        }
        // Update ntc and cleaned count
        rx_ring->next_to_clean = ((rx_ring->next_to_clean+1) < rx_ring->count) ? (rx_ring->next_to_clean+1) : 0;
        cleaned_count++;
    } while (likely(total_rx_packets < budget));

    if (cleaned_count)
        ixgbe_alloc_rx_buffers(rx_ring, cleaned_count);

    // TODO: Currently not handling LRO packets
//#ifndef IXGBE_NO_LRO
    //  ixgbe_lro_flush_all(q_vector);
//#endif /* IXGBE_NO_LRO */

    return total_rx_packets;
}

/******************************************************/
/************************* TX *************************/
/******************************************************/

int setup_gpu_tx_env(struct ixgbe_adapter *adapter){

    struct net_device *netdev = adapter->netdev;
    struct ixgbe_ring *tx_ring;
    ubc_gpu_hooks *hook;
    unsigned i = 0;

    hook = netdev->hook_to_gpu_mod; // Get hook to GNoM

    // Check if the GNoM TX flow is configured, set number of available GTXBs, and register GNoM TX function
    if(!hook  || !hook->is_gpu_tx_ready(&gtx_ring_config.num_gpu_buffers, &gtxb.bufs, &ixgbe_gnom_tx_send)){
        if(!hook) printk("[GNoM_ixgbe]: Failed Hook...(hook = %p)\n", hook);
        else printk("[GNoM_ixgbe]: GPU TX buffers not set yet! (hook = %p)\n", hook);
        goto err;
    }

    // If no buffers returned, error out
    if(gtx_ring_config.num_gpu_buffers <= 0){
        printk("[GNoM_ixgbe]: Number of GTX buffers incorrect...\n");
        goto err;
    }

    printk("[GNoM_ixgbe]: GTXB.bufs = %p\n", gtxb.bufs);
    printk("[GNoM_ixgbe]: GNoM TX up (%d GTXBs)\n", gtx_ring_config.num_gpu_buffers);

    gtxb.req_buf_ind = 0;
    gtxb.req_batch_ind = 0;

    // GPU module is up and running, populate GPU buffers, initialize structures
    gtx_ring.g_ring_bufs = (nic_gpu_ring_buf *)vmalloc((gtx_ring_config.num_gpu_buffers+1)*sizeof(nic_gpu_ring_buf));         // Storage for TX buffers structures
    gtx_ring.g_ring_free_bufs = (nic_gpu_ring_buf **)vmalloc((gtx_ring_config.num_gpu_buffers+1)*sizeof(nic_gpu_ring_buf *)); // List of free TX buffers

    // If memory allocation for GTXRings fail, error out
    if(!gtx_ring.g_ring_bufs || !gtx_ring.g_ring_free_bufs)
        goto gpu_alloc_err;

    // Configure head and tail pointers for GTXB management
    gtx_ring.free_head = gtx_ring_config.num_gpu_buffers;   // Next index to push to
    gtx_ring.free_tail = 0;                                 // Next index to pull from

    for(i=0; i<gtx_ring_config.num_gpu_buffers; ++i){
        gtx_ring.g_ring_bufs[i].bid = i;                // Set BID
        gtx_ring.g_ring_bufs[i].b_state = GNOM_FREE;    // Set state to free

        // Set metadata for this GTXB
        if(hook->gpu_set_buffers(&gtx_ring.g_ring_bufs[i].m_page_addr,          // Virtual CPU mmaped page address
                                    &gtx_ring.g_ring_bufs[i].m_gpu_host_addr,   // CUDA virtual address
                                    &gtx_ring.g_ring_bufs[i].dma,               // CUDA physical address
                                    &gtx_ring.g_ring_bufs[i].page_offset,       // Page offset
                                    i,                                          // BID
                                    GNOM_TX))                                   // TX buffer
            goto gpu_alloc_err;

        // Initially fill free list with all available buffers
        gtx_ring.g_ring_free_bufs[i] = &gtx_ring.g_ring_bufs[i];
    }

    // There will be at least two TX rings. One for regular CPU traffic, one for GNoM traffic.
    // Set the last ring for GPU.
    if(adapter->num_tx_queues > 1){
        tx_ring = adapter->tx_ring[adapter->num_tx_queues - 1];
        tx_ring->is_gpu_ring = 1; // Set this ring as a GPU ring

        gnom_tx_ring = tx_ring; // Update gnom_tx_ring to point to the single GPU ring
        gnom_tx_ring->next_to_clean = 0;
        gnom_tx_ring->next_to_use = 0;
    }else{
        /* Error: No TX ring available for GNoM */
        printk("[GNoM_ixgbe]: Error: No TX ring available for GNoM - Need at least 1 CPU and 1 GPU ring\n");
        goto gpu_alloc_err;
    }

    // Set the enable flag for GTX
    gtx_ring_config.is_set = 1;

    printk("[GNoM_ixgbe]: \t%d GPU TX buffers successfully initialized in NIC driver on %s\n", gtx_ring_config.num_gpu_buffers, netdev->name);

    rmb(); // Ensure all buffers are initialized before registering them with the NIC

    return 0;

gpu_alloc_err:
    printk("[GNoM_ixgbe]: Error: Can't allocate the %d'th GPU buffer...\n", i);

    gnom_cleanup_tx();

err:
    for(i=0; i<adapter->num_tx_queues; ++i){
        tx_ring = adapter->tx_ring[i];
        tx_ring->is_gpu_ring = 0; // Set this ring as a CPU ring again
    }

    return 1;

}

bool ixgbe_recycle_gtxb(int batch_id, int buf_id){
    // For TX, we recycle buffers one at a time on cleaning the TX IRQ
    // but allocate buffers in batches.
    // Number of free spots == total number of buffers, so no need to check if head == tail

    int bid;

    bid = gtxb.buf_ids[batch_id][buf_id];
    if(unlikely(bid == -1)) return false;

    gtx_ring.g_ring_free_bufs[gtx_ring.free_head] = &gtx_ring.g_ring_bufs[bid];
    gtx_ring.g_ring_free_bufs[gtx_ring.free_head]->b_state = GNOM_FREE;

    // Update gtx_ring.free_head (gtx_ring_config.num_gpu_buffers+1 locations)
    gtx_ring.free_head = ((gtx_ring.free_head+1) < gtx_ring_config.num_gpu_buffers) ? (gtx_ring.free_head + 1) : 0;

    gtxb.buf_ids[batch_id][buf_id] = -1;
    gnom_inc_num_tx_recycled(1);

    return true;
}

// Pull new buffer off of the free gpu buffer list, pop from head
// Returns index of buffer batch on success, -1 on error
int gnom_fetch_tx_buffer_batch(int num_req){
    int i;
    int batch_ind;
    nic_gpu_ring_buf* ring_buf;

    batch_ind = gtxb.req_batch_ind; // Select batch index to pass to CPU daemon thread
    for(i=0; i<num_req; ++i){

        if(unlikely(gtx_ring.free_tail == gtx_ring.free_head )){ // No free buffers left :(. This will cause stalling on TX. 
            printk("[GNoM_ixgbe]: Error: tail(%d) == head (%d)\n", gtx_ring.free_tail, gtx_ring.free_head);
            return -1;
        }

        ring_buf = gtx_ring.g_ring_free_bufs[gtx_ring.free_tail];   // Set buffer
        gtx_ring.g_ring_free_bufs[gtx_ring.free_tail] = NULL;       // Remove from free list

        // Update grx_ring.free_tail (grx_ring_config.num_gpu_buffers+1 locations)
        gtx_ring.free_tail = ((gtx_ring.free_tail+1) < gtx_ring_config.num_gpu_buffers) ? (gtx_ring.free_tail + 1) : 0;

        if(unlikely(ring_buf->b_state != GNOM_FREE)){
            printk("[GNoM_ixgbe]: Error: GRing buffer state not free: %d\n", ring_buf->b_state);
            return -2;
        }

        ring_buf->b_state = GNOM_ON_GPU;

        // Set buffer id index and CUDA buffer pointer
        gtxb.buf_ids[batch_ind][i] = ring_buf->bid;
        gtxb.bufs[batch_ind*NUM_RX_QUEUE_BUF + i] = ring_buf->m_gpu_host_addr + ring_buf->page_offset;

    }
    gnom_inc_num_tx_fetched(num_req);
    // Increment batch_ind to next batch
    gtxb.req_batch_ind = (gtxb.req_batch_ind + 1) % NUM_RX_QUEUE_BUF;

    return batch_ind;
}

int ixgbe_gnom_tx_send(int batch_ind, int buf_ind, unsigned int size){
    // Should be called with batch_id and buf_id, all required information is stored in the gtxb structure
    /*
        - Flags == 0 when set to tx_ring initially
        - TSO is called, but bails out on first exit cond
        - TX_CSUM is called
        - IP_SUMMED seems to be CHECKSUM_PARTIAL
        - Update TX checksum
        - Then ctxdesc is called
    */
    nic_gpu_ring_buf* tx_ring_buf;
    int tx_bid;
    struct ixgbe_tx_buffer *tx_buffer;
    union ixgbe_adv_tx_desc *tx_desc;
    void *send_buf;

    // TODO: Need to implement the higher level flags
    u32 tx_flags = 0; // first->tx_flags;            
    u32 cmd_type = 0; // ixgbe_tx_cmd_type(tx_flags);
    u16 i = 0;

    if(unlikely(!gnom_tx_ring->is_gpu_ring)){
        printk("[GNoM_ixgbe]: Error - Calling ixgbe_gnom_tx_send with non-GPU RX ring\n");
        return -1;
    }

    // Grab specified GRXB ring buffer from list
    tx_bid = gtxb.buf_ids[batch_ind][buf_ind];
    if(unlikely(tx_bid == -1)) return -1;

    tx_ring_buf = &gtx_ring.g_ring_bufs[tx_bid];


    /*
    // Debugging to check if the TX packet is coming from GNoM
    printk("[GNoM_ixgbe]: (%d)\n", buf_ind);
    if(likely(verify_pkt(tx_ring_buf->m_page_addr))){
        print_pkt_hdr(tx_ring_buf->m_page_addr, 0);
    }else{
        printk("\tIncorrect packet\n");
    }
    return 0;
    */

    if(unlikely(!verify_pkt(tx_ring_buf->m_page_addr)) ){
        return -1;
    }

    // Set buffer state to on NIC
    tx_ring_buf->b_state = GNOM_ON_NIC;

    // Grab the actual buffer containing packet and packet header
    send_buf = (void *)(tx_ring_buf->m_page_addr + tx_ring_buf->page_offset);

    /* record the location of the first descriptor for this packet */
    tx_buffer = &gnom_tx_ring->tx_buffer_info[gnom_tx_ring->next_to_use]; // Select next TX buffer
    tx_buffer->skb = NULL;                                      // No SKB for GNoM
    tx_buffer->bytecount = size;                                // Set length
    tx_buffer->gso_segs = 1;                                    // Single GSO segment
    tx_buffer->tx_flags = tx_flags;


    /* GNoM record length, and DMA address */
    dma_unmap_len_set(tx_buffer, len, size);
    dma_unmap_addr_set(tx_buffer, dma, tx_ring_buf->dma);
    tx_buffer->gpu_page = virt_to_page(tx_ring_buf->m_page_addr);
    tx_buffer->gpu_page_offset = tx_ring_buf->page_offset;
    tx_buffer->gpu_batch_id = batch_ind;
    tx_buffer->gpu_buffer_id = buf_ind;

    tx_buffer->protocol = gnom_eth_proto(send_buf);

    ixgbe_gnom_tx_csum(gnom_tx_ring, tx_buffer, send_buf);

    i = gnom_tx_ring->next_to_use;

    // Grab the descriptor for this gnom_tx_ring
    tx_desc = IXGBE_TX_DESC(gnom_tx_ring, i);
    ixgbe_tx_olinfo_status(tx_desc, tx_buffer->tx_flags, size /*skb->len - hdr_len*/);
    cmd_type = ixgbe_tx_cmd_type(tx_flags);

    dma_sync_single_range_for_device(gnom_tx_ring->dev,
                                    dma_unmap_addr(tx_buffer, dma),
                                    tx_buffer->gpu_page_offset,
                                    size,
                                    DMA_TO_DEVICE);

    tx_desc->read.buffer_addr = cpu_to_le64(dma_unmap_addr(tx_buffer, dma));


    /* write last descriptor with RS and EOP bits */
    cmd_type |= size | IXGBE_TXD_CMD;
    tx_desc->read.cmd_type_len = cpu_to_le32(cmd_type);

//    netdev_tx_sent_queue(netdev_get_tx_queue(gnom_tx_ring->netdev,
//                                             gnom_tx_ring->queue_index),
//                         size /*first->bytecount*/);

    /* set the timestamp */
    tx_buffer->time_stamp = jiffies;

    /*
     * Force memory writes to complete before letting h/w know there
     * are new descriptors to fetch.  (Only applicable for weak-ordered
     * memory model archs, such as IA-64).
     *
     * We also need this memory barrier to make certain all of the
     * status bits have been updated before next_to_watch is written.
     */
    wmb();

    /* set next_to_watch value indicating a packet is present */
    tx_buffer->next_to_watch = tx_desc;

    //printk("[GNoM_ixgbe] SEND: ring: %p, tx_buffer: %p, tx_desc: %p, next_to_use: %d, next_to_clean: %d, next_to_watch: %p\n", gnom_tx_ring, tx_buffer, tx_desc, gnom_tx_ring->next_to_use, gnom_tx_ring->next_to_clean, tx_buffer->next_to_watch);

    i++;
    if (i == gnom_tx_ring->count)
        i = 0;

    gnom_tx_ring->next_to_use = i;

    /*
    // Testing the latency that the TX buffer is on the NIC 
    tx_buffer->sent_time = ktime_get();
    tx_buffer->sanity_test_val = sanity_val++;
    */

    /* notify HW of packet */
    writel(i, gnom_tx_ring->tail);

    /*
     * we need this if more than one processor can write to our tail
     * at a time, it synchronizes IO on IA64/Altix systems
     */
    mmiowb();

    return 0;

}

// GNoM TX CLEAN IRQ
bool ixgbe_clean_gpu_tx_irq(struct ixgbe_q_vector *q_vector,
                               struct ixgbe_ring *tx_ring)
{
    struct ixgbe_adapter *adapter = q_vector->adapter;
    struct ixgbe_tx_buffer *tx_buffer;
    union ixgbe_adv_tx_desc *tx_desc;
    //unsigned int total_bytes = 0, total_packets = 0;
    unsigned int budget = q_vector->tx.work_limit;
    //unsigned int budget = 4096;
    unsigned int i = tx_ring->next_to_clean;
    int count=0;

    if (test_bit(__IXGBE_DOWN, &adapter->state))
        return true;

    tx_buffer = &tx_ring->tx_buffer_info[i];
    tx_desc = IXGBE_TX_DESC(tx_ring, i);
    i -= tx_ring->count;

    do {
        union ixgbe_adv_tx_desc *eop_desc = tx_buffer->next_to_watch;

        /* if next_to_watch is not set then there is no work pending */
        if (!eop_desc)
            break;

        count++;
        /* prevent any other reads prior to eop_desc */
        read_barrier_depends();

        /* if DD is not set pending work has not been completed */
        if (!(eop_desc->wb.status & cpu_to_le32(IXGBE_TXD_STAT_DD)))
            break;

        /* clear next_to_watch to prevent false hangs */
        tx_buffer->next_to_watch = NULL;

        // Removing the "dev_kfree_skb_any" code

        dma_sync_single_range_for_cpu(tx_ring->dev,
                          dma_unmap_addr(tx_buffer, dma),
                          tx_buffer->gpu_page_offset,
                          dma_unmap_len(tx_buffer, len),
                          DMA_TO_DEVICE);
        
        // Adding GNoM TX buffer recycle
        if(unlikely(!ixgbe_recycle_gtxb(tx_buffer->gpu_batch_id, tx_buffer->gpu_buffer_id))){
            printk("[INTEL_GNOM]: ixgbe_recycle_gtxb failed...\n");
            return true;
        }
        tx_total_count++;
        
        /*
        // Testing the latency that the TX packet is on the NIC
        tx_total_time += ktime_to_ns(ktime_sub(ktime_get(), tx_buffer->sent_time));
        sanity_sum += tx_buffer->sanity_test_val;
        */

        /* clear tx_buffer data */
        dma_unmap_len_set(tx_buffer, len, 0);

        // printk("[GNoM_ixgbe] CLEAN: ring: %p, tx_buffer: %p, tx_desc: %p, next_to_use: %d, next_to_clean: %d, next_to_watch: %p\n", tx_ring, tx_buffer, tx_desc, tx_ring->next_to_use, tx_ring->next_to_clean, tx_buffer->next_to_watch);

        /* unmap remaining buffers */
        while (unlikely(tx_desc != eop_desc)) {
            tx_buffer++;
            tx_desc++;
            i++;
            if (unlikely(!i)) {
                i -= tx_ring->count;
                tx_buffer = tx_ring->tx_buffer_info;
                tx_desc = IXGBE_TX_DESC(tx_ring, 0);
            }

            /* unmap any remaining paged data */
            // Shouldn't be here in GNoM TX testing
            if (dma_unmap_len(tx_buffer, len)) {
                printk("[INTEL_GNOM]: In unmapping remaining buffers code... Shouldn't be here in GNoM TX testing\n");
                printk("Next to clean: %d, next_to_watch: %p\n", tx_ring->next_to_clean, tx_buffer->next_to_watch);
                break;
            }
        }

        /* move us one more past the eop_desc for start of next pkt */
        tx_buffer++;
        tx_desc++;
        i++;
        if (unlikely(!i)) {
            i -= tx_ring->count;
            tx_buffer = tx_ring->tx_buffer_info;
            tx_desc = IXGBE_TX_DESC(tx_ring, 0);
        }

        /* issue prefetch for next Tx descriptor */
        prefetch(tx_desc);

        /* update budget accounting */
        budget--;
    } while (likely(budget));

    //printk("[GNoM_ixgbe]: TX_clean count: %d\n", count);

    i += tx_ring->count;
    tx_ring->next_to_clean = i;

    // Original TX code below. Initially testing GNoM TX with minimal required processing. 
/*
    netdev_tx_completed_queue(netdev_get_tx_queue(tx_ring->netdev,
                                                  tx_ring->queue_index),
                              total_packets, total_bytes);
*/
//
//#define TX_WAKE_THRESHOLD (DESC_NEEDED * 2)
//    if (unlikely(total_packets && netif_carrier_ok(netdev_ring(tx_ring)) &&
//                 (ixgbe_desc_unused(tx_ring) >= TX_WAKE_THRESHOLD))) {
//        /* Make sure that anybody stopping the queue after this
//         * sees the new next_to_clean.
//         */
//        smp_mb();
//#ifdef HAVE_TX_MQ
//        if (__netif_subqueue_stopped(netdev_ring(tx_ring),
//                                     ring_queue_index(tx_ring))
//            && !test_bit(__IXGBE_DOWN, &q_vector->adapter->state)) {
//            netif_wake_subqueue(netdev_ring(tx_ring),
//                                ring_queue_index(tx_ring));
//            ++tx_ring->tx_stats.restart_queue;
//        }
//#else
//        if (netif_queue_stopped(netdev_ring(tx_ring)) &&
//            !test_bit(__IXGBE_DOWN, &q_vector->adapter->state)) {
//            netif_wake_queue(netdev_ring(tx_ring));
//            ++tx_ring->tx_stats.restart_queue;
//        }
//#endif
//    }

    return !!budget;
}

int verify_pkt(void *data){
    udp_header *uh = (udp_header *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
    if(ntohs(uh->source) == 9960){
        return 1;
    }else{
        return 0;
    }
}

/******************************************************/
/************************ MISC ************************/
/******************************************************/

int test_tx(void *data, dma_addr_t dma){

    struct ixgbe_ring *tx_ring = NULL;
    struct ixgbe_tx_buffer *tx_buffer;
    union ixgbe_adv_tx_desc *tx_desc;

    int size = 1024;
    u32 tx_flags = 0; // first->tx_flags;             // TODO: Need to implement the higher level flags
    u32 cmd_type = 0; // ixgbe_tx_cmd_type(tx_flags);
    u16 i = 0;


    if(tx_ring == NULL){
        if(!gnom_tx_adapter) return -1;
        tx_ring = gnom_tx_adapter->tx_ring[1];
    }

    /* record the location of the first descriptor for this packet */
    tx_buffer = &tx_ring->tx_buffer_info[tx_ring->next_to_use]; // Select next TX buffer
    tx_buffer->skb = NULL;                                      // No SKB for GNoM
    tx_buffer->bytecount = size;                                // Set length
    tx_buffer->gso_segs = 1;                                    // Single GSO segment
    tx_buffer->tx_flags = tx_flags;

    tx_buffer->gpu_buffer_id = 123; // Check for 123 on clean TX irq

    /* GNoM record length, and DMA address */
    dma_unmap_len_set(tx_buffer, len, size);
    dma_unmap_addr_set(tx_buffer, dma, dma);


    tx_buffer->protocol = gnom_eth_proto(data);

    ixgbe_gnom_tx_csum(tx_ring, tx_buffer, data);

    i = tx_ring->next_to_use;

    // Grab the descriptor for this tx_ring
    tx_desc = IXGBE_TX_DESC(tx_ring, i);

    ixgbe_tx_olinfo_status(tx_desc, tx_buffer->tx_flags, size );

    cmd_type = ixgbe_tx_cmd_type(tx_flags);


    dma_sync_single_range_for_device(tx_ring->dev,
                                    dma_unmap_addr(tx_buffer, dma),
                                    0,
                                    1024,
                                    DMA_TO_DEVICE);


    tx_desc->read.buffer_addr = cpu_to_le64(dma_unmap_addr(tx_buffer, dma));


    /* write last descriptor with RS and EOP bits */
    cmd_type |= size | IXGBE_TXD_CMD;
    tx_desc->read.cmd_type_len = cpu_to_le32(cmd_type);

    netdev_tx_sent_queue(netdev_get_tx_queue(tx_ring->netdev,
                                             tx_ring->queue_index),
                         size /*first->bytecount*/);

    /* set the timestamp */
    tx_buffer->time_stamp = jiffies;

    /*
     * Force memory writes to complete before letting h/w know there
     * are new descriptors to fetch.  (Only applicable for weak-ordered
     * memory model archs, such as IA-64).
     *
     * We also need this memory barrier to make certain all of the
     * status bits have been updated before next_to_watch is written.
     */
    wmb();

    /* set next_to_watch value indicating a packet is present */
    tx_buffer->next_to_watch = tx_desc;

    printk("[GNoM_ixgbe]: SEND: ring: %p, tx_buffer: %p, tx_desc: %p, next_to_use: %d, next_to_clean: %d, next_to_watch: %p\n", tx_ring, tx_buffer, tx_desc, tx_ring->next_to_use, tx_ring->next_to_clean, tx_buffer->next_to_watch);

    i++;
    if (i == tx_ring->count)
        i = 0;

    tx_ring->next_to_use = i;

    /* notify HW of packet */
    writel(i, tx_ring->tail);

    /*
     * we need this if more than one processor can write to our tail
     * at a time, it synchronizes IO on IA64/Altix systems
     */
    mmiowb();

    return 0;

}



#ifdef GNOM_DEBUG
void print_pkt_hdr(void *data, int is_tcp){
    unsigned i=0;
    ether_header *eh = (ether_header *)data;
    ip_header *iph = (ip_header *)((size_t)data + sizeof(ether_header));
    struct tcphdr *tcph = NULL;
    udp_header *uh = NULL;

    printk("Packet header contents: \n");

    /***** ETHERNET HEADER *****/
    printk("\t==Ethernet header==\n");
    printk("\t\tDest: ");
    for(i=0; i<ETH_ALEN; ++i)
        printk("%hhx ", eh->ether_dhost[i]);
    printk("\n\t\tSource: ");
    for(i=0; i<ETH_ALEN; ++i)
        printk("%hhx ", eh->ether_shost[i]);
    printk("\n\t\tType: %hx\n", eh->ether_type);
    /***** END ETHERNET HEADER *****/

    /***** IP HEADER *****/
    printk("\t==IP header==\n");
    printk("\t\tVersion+hdr_len: %hhu\n", iph->version);
    printk("\t\tTOS: %hhu\n", iph->tos);
    printk("\t\tTotal Length: %hu\n", ntohs(iph->tot_len));
    printk("\t\tID: %hu\n", ntohs(iph->id));
    printk("\t\tFrag_off: %hu\n", iph->frag_off);
    printk("\t\tTTL: %hhu\n", iph->ttl);
    printk("\t\tProtocol: %hhu\n", iph->protocol);
    printk("\t\tchecksum: %hu\n", ntohs(iph->check));
    printk("\t\tSource address: %x\n", ntohl(iph->saddr));
    printk("\t\tDest address: %x\n", ntohl(iph->daddr));
    /***** END IP HEADER *****/

    if(is_tcp){
        /***** TCP HEADER *****/
        tcph = (struct tcphdr *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
        printk("\t\tSource port: %hu\n", ntohs(tcph->source));
        printk("\t\tDest port: %hu\n", ntohs(tcph->dest));
        printk("\t\tSEQ num: %u\n", ntohl(tcph->seq));
        printk("\t\tACK num: %u\n", ntohl(tcph->ack_seq));

        printk("\t\tres: %hu\n", tcph->res1);
        printk("\t\tdoff: %hu\n", tcph->doff);

        printk("\t\tfin: %hu\n", tcph->fin);
        printk("\t\tsyn: %hu\n", tcph->syn);
        printk("\t\trst: %hu\n", tcph->rst);
        printk("\t\tpsh: %hu\n", tcph->psh);
        printk("\t\tack: %hu\n", tcph->ack);
        printk("\t\turg: %hu\n", tcph->urg);
        printk("\t\tece: %hu\n", tcph->ece);
        printk("\t\tcwr: %hu\n", tcph->cwr);

        printk("\t\twindow: %hu\n", tcph->window);
        printk("\t\tcheck: %hu\n", tcph->check);
        printk("\t\turg_ptr: %hu\n", tcph->urg_ptr);

        /***** END TCP HEADER *****/
    }else{
        /***** UDP HEADER *****/
        uh = (udp_header *)((size_t)data + sizeof(ether_header) + sizeof(ip_header));
        printk("\t==UDP header==\n");
        printk("\t\tSource port: %hu\n", ntohs(uh->source));
        printk("\t\tDest port: %hu\n", ntohs(uh->dest));
        printk("\t\tLength: %hu\n", ntohs(uh->len));
        printk("\t\tChecksum: %hu\n", uh->check);
        /***** END UDP HEADER *****/
    }


}

void print_skb_info(struct sk_buff *skb){
    printk("\n\tskb->data: %p\n\tskb->len: %d\n\tsize: %d\n\tdata_len: %d\n\tnfrags: %d\n",
            skb->data, skb->len, skb_headlen(skb), skb->data_len, skb_shinfo(skb)->nr_frags);

}
#endif
