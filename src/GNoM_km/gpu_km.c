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
 *  gpu_km.c
*/

#include "gpu_km.h"

// If DEBUG_PRINT is defined (SLOW), all print messages enabled on the runtime path (e.g., when packets are recieved/sent). 
// Used to trace packet batches through GNoM-KM.
//#define DEBUG_PRINT

/************************************************************************/
// Function Declarations
/************************************************************************/
static int device_open(struct inode *, struct file *);
static int device_release(struct inode *, struct file *);
static ssize_t device_write(struct file *filp, const char __user *buf, size_t count, loff_t * ppos);
static long device_ioctl(struct file *, unsigned int cmd, unsigned long arg);
static ssize_t device_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos);
static int device_mmap(struct file *filp, struct vm_area_struct *vma);

// IXGBE RPC functions
static bool (* ixgbe_callback_recycle_grxb_batch)(int batch_ind, int num_req);  // Recycle batch of GRX buffers
static int (* ixgbe_callback_register)(int cmd);                                // Used to issue commands to the IXGBE driver
static int (* gnom_tx)(int batch_ind, int buf_ind, unsigned int size);          // TX Send function


// NVIDIA GPUDirect callback function
void free_callback(void *data);

static int set_buffers(void **gpu_buf, uint64_t *cuda_addr, void *dma, int *page_offset, int bid, int is_tx);
static int is_gpu_rx_ready(int *num_buf, uint64_t **RX_buffer_pointers,
        bool (*recycle_grxb_batch)(int batch_ind, int num_req), int (*ixgbe_reg)(int cmd));
static int is_gpu_tx_ready(int *num_buf, uint64_t **TX_buffer_pointers, int (*ixgbe_gnom_tx)(int batch_ind, int buf_ind, unsigned int size));
static int dispatch_work(int rx_b_ind, int tx_b_ind, int num_req);

int get_dma_info(struct page **kernel_page, void **kernel_va_addr, int set_finished);
int set_device(struct device *dev);

void mmap_open(struct vm_area_struct *vma);
void mmap_close(struct vm_area_struct *vma);
static int mmap_fault(struct vm_area_struct *vma, struct vm_fault *vmf);
static int gpu_notifier(struct notifier_block *this, unsigned long msg, void *data);

/************************************************************************/
/************************************************************************/

/************************************************************************/
// Data structures
/************************************************************************/
static struct file_operations gpu_km_ops = {
    .owner          = THIS_MODULE,
    .open           = device_open,
    .release        = device_release,
    .write          = device_write,
    .read           = device_read,
    .unlocked_ioctl = device_ioctl,
    .mmap           = device_mmap
};

struct vm_operations_struct mmap_vm_ops = {
    .open = mmap_open,
    .close = mmap_close,
    .fault = mmap_fault,
};

static struct notifier_block gpu_netdev_notifier = {
  .notifier_call = gpu_notifier,
};

static ubc_gpu_hooks gpu_hooks = {
        .magic = UBC_GPU,
        .gpu_set_buffers = set_buffers,
        .is_gpu_rx_ready = is_gpu_rx_ready,
        .is_gpu_tx_ready = is_gpu_tx_ready,
        .gpu_dispatch_work = dispatch_work,
        .get_dma_info = get_dma_info,
        .gpu_set_device = set_device,
};
/************************************************************************/
/************************************************************************/


/************************************************************************/
// Globals
// TODO: Cleanup any old unused variables.
/************************************************************************/


static struct class *gpu_km_class = NULL;    // Creating a class to contain our module

static int is_send_finished = 0;
static int IS_DONE_FLAG = 0; // Flag to signal completion of the host process.

/****** Stat counters *******/
static int num_drop_batches = 0;
static int num_recv_batches = 0;
static int num_host_thread_waits = 0;
/*********************************/

static struct semaphore nic_gpu_lock;
static struct semaphore user_gpu_lock;

static lw_network_requests *m_lw_pending_batches = NULL;
static volatile int pending_batch_head = 0;
static volatile int pending_batch_tail = 0;


static kernel_args_hdr m_kernel_args;
static nvidia_p2p_page_table_t **page_table_rx = NULL;
static nvidia_p2p_page_table_t **page_table_tx = NULL;

static int pkt_rcv_cnt = 0;

// RX buffer pointers mmap'ed
static uint64_t *current_RX_buffer_pointers = NULL;
static uint64_t *current_TX_buffer_pointers = NULL;

struct device *gnom_dev = NULL;

static DECLARE_WAIT_QUEUE_HEAD(m_req_wq);	// Create a request wait queue

static int gpu_rx_ready = 0;
static int gpu_tx_ready = 0;

// User specifies number of rings and number of buffers per ring
// RX
static int num_gpu_rx_pages = 0;
static int num_gpu_rx_buf = 0;
static int num_gpu_rx_buf_per_page = 0;
static int num_gpu_rx_bins = 0;

// TX
static int num_gpu_tx_pages = 0;
static int num_gpu_tx_buf = 0;
static int num_gpu_tx_buf_per_page = 0;
static int num_gpu_tx_bins = 0;

static gpu_buf_t *m_gpu_rx_bufs = NULL; 
static gpu_buf_t *m_gpu_tx_bufs = NULL;

static struct page *su_page = NULL;
static char *sk_address = NULL;
static char *global_mmaped_ptr = NULL;

static unsigned long long total_time = 0;
static unsigned long long total_count = 0;


void gnom_print(const char *msg){
    printk("[GNoM_km]: %s", msg);
}

void print_gpu_buf_t(gpu_buf_t *buf, int buf_id){
    printk("[GNoM_km]: (Buffer %d) cuda_va: %p, host_va: %p, user_pg: %p, dma: %p, offset: %u\n",
            buf_id,
            (void *)buf->cuda_addr,
            buf->host_addr,
            buf->user_pg,
            (void *)buf->dma,
            buf->page_offset);
}

void mmap_open(struct vm_area_struct *vma){
    struct mmap_info *info = (struct mmap_info *)vma->vm_private_data;
    printk("[GNoM_km]: Mmap_open called...\n");
	info->reference_cnt++;
}

void mmap_close(struct vm_area_struct *vma){
    struct mmap_info *info = (struct mmap_info *)vma->vm_private_data;
 	struct page *page;
    int i;
    printk("[GNoM_km]: Mmap_closed called...\n");

#ifdef DO_GNOM_TX
    for(i=0; i<1024; ++i){ // Unpin all the pages
#else
    for(i=0; i<512; ++i){ // Unpin all the pages
#endif
        page = virt_to_page((size_t)info->data + (i*CPU_PAGE_SIZE));

        if(PageReserved(page)){
            ClearPageReserved(page);
        }
    }
    info->reference_cnt--;
}

static int device_mmap(struct file *filp, struct vm_area_struct *vma){

    printk("[GNoM_km]: device_mmap called...\n");
    vma->vm_ops = &mmap_vm_ops;
    vma->vm_flags |= VM_RESERVED;
    vma->vm_private_data = filp->private_data;
    mmap_open(vma);
    
    return 0;
}

static int mmap_fault(struct vm_area_struct *vma, struct vm_fault *vmf){

	struct page *page;
	struct mmap_info *info = (struct mmap_info *)vma->vm_private_data;

#ifdef DEBUG_PRINT
	printk("[GNoM_km]: mmap_fault called...(page offset: %lu, total offset: %lu)\n", vmf->pgoff, vmf->pgoff << PAGE_SHIFT);
#endif 

	if(!info->data){
		printk("[GNoM_km]: Error in mmap_fault, no data...\n");
		return -1;
	}

	page = virt_to_page(info->data + (vmf->pgoff << PAGE_SHIFT));
	get_page(page);

    // Reserve the page
    SetPageReserved(page);

    // Return the page
	vmf->page = page;

#ifdef DEBUG_PRINT 
	printk("[GNoM_km]: page = %p\n", page);
#endif

	return 0;
}

static int device_open(struct inode *inode, struct file *filp){
	struct mmap_info *info;

#ifdef DO_GNOM_TX
	int off=1;
#endif

	printk ("device_open: %d.%d\n", MAJOR (inode->i_rdev), MINOR (inode->i_rdev));

	IS_DONE_FLAG = 0;
	info = kmalloc(sizeof(struct mmap_info), GFP_KERNEL);
	info->data = (char *)__get_free_pages(GFP_KERNEL, K_PAGE_ORDER); // 2^K_PAGE_ORDER pages

    if(!info->data){
        printk("[GNoM_km]: Insufficient memory to allocate shared pages...\n");
        return -ENOMEM;
    }

    global_mmaped_ptr = info->data;
    
    printk("[GNoM_km]: GPU-NoM dev_open mapped data: %p\n",  info->data);
	memcpy(info->data, "GNoM: ", 32);

	filp->private_data = info;
    current_RX_buffer_pointers = (uint64_t *) info->data; // First half of buffer

#ifdef DO_GNOM_TX
    off = 1<<(K_PAGE_ORDER-1);
    current_TX_buffer_pointers = (uint64_t *)((uint64_t)info->data + (off*CPU_PAGE_SIZE) ); // Second half of buffer
#endif

	return 0;
}

static int device_release(struct inode *inode, struct file *filp){
    struct mmap_info *info=NULL;
    struct page *page=NULL;
	unsigned i=0;
	unsigned num_pages=0;

	pr_info("device_release: %d.%d\n", MAJOR (inode->i_rdev), MINOR (inode->i_rdev));

	IS_DONE_FLAG = 1;

	// Free RX buffer pointer queues
	info = filp->private_data;
	if (info->data ) {
		printk("[GNoM_km] on release: (%d) <%p>\n", info->reference_cnt, info->data);

		num_pages = (1<<K_PAGE_ORDER);

        for(i=0; i<num_pages; ++i){ // Unpin all of the the allocated pages
			page = virt_to_page((size_t)info->data + (i*CPU_PAGE_SIZE));
			if(PageReserved(page)){
                ClearPageReserved(page);
            }
		}

		free_pages((unsigned long)info->data, K_PAGE_ORDER); // Free 2^K_PAGE_ORDER pages
		kfree(info);
		filp->private_data = NULL;
    }

	printk("[GNoM_km]: %u mmaped Pages succesfully released\n", i);

	return 0;
}

// This function wakes up any sleeping processes waiting for work to launch on the GPU
// req_buffers contains a list of pointers to GPU buffers containing pkts, which is used as
// an argument to the CUDA kernel. 1 request per buffer.
static int batch_id = 0;

static int dispatch_work(int rx_b_ind, int tx_b_ind, int num_req){

	int nh = (pending_batch_head < (MAX_BATCH_QUEUE-1)) ? (pending_batch_head + 1) : 0;

	if( unlikely(nh == pending_batch_tail) ){ // If our next head happens to
		// Don't try and add if we're out of space... just error out and drop the batch
		num_drop_batches++;
		goto dispatch_err;
	}

	num_recv_batches++;

	m_lw_pending_batches[pending_batch_head].batch_id = batch_id++;         // Sanity check on batch recycle
	m_lw_pending_batches[pending_batch_head].num_reqs = num_req;            // Number of requests in this batch
	m_lw_pending_batches[pending_batch_head].rx_batch_ind = rx_b_ind;       // Index into mmap'ed RXB pointer buffer
#ifdef DO_GNOM_TX
	m_lw_pending_batches[pending_batch_head].tx_batch_ind = tx_b_ind;       // Index into mmap'ed TXB pointer buffer
#endif

	pending_batch_head = nh;

#ifdef DEBUG_PRINT
	printk("[GNoM_km]: Dispatch work (recv_batch: %d, head: %d, tail: %d)\n", num_recv_batches, pending_batch_head, pending_batch_tail);
#endif

	wake_up(&m_req_wq);

	return 0;

dispatch_err:
	printk("[GNoM_km]: Error: Could not dispatch buffer, no space in the queue 	...\n");
	return 1;
}

static int set_buffers(void **gpu_buf, uint64_t *cuda_addr, void *dma, int *page_offset, int bid, int rx_or_tx){

    gpu_buf_t *gpu_bufs;

    if(unlikely(bid < 0))
        goto set_buf_err;

    // Select from the correct pool of buffers to set. RX = 0, TX = 1
    if(rx_or_tx == GNOM_RX){
        if(unlikely(bid >= num_gpu_rx_buf)){
            printk("[GNoM_km]: Error: bid (%d) > num_gpu_rx_buf (%d)\n", bid, num_gpu_rx_buf);
            goto set_buf_err;
        }
        gpu_bufs = m_gpu_rx_bufs;
    }else{
        if(unlikely(bid >= num_gpu_tx_buf)){
            printk("[GNoM_km]: Error: bid (%d) > num_gpu_tx_buf (%d)\n", bid, num_gpu_tx_buf);
            goto set_buf_err;
        }
        gpu_bufs = m_gpu_tx_bufs;
    }
    
    if(!gpu_bufs[bid].host_addr || !gpu_bufs[bid].dma){
        printk("[GNoM_km]: Error: bid = %d: host_addr %p or dma %p not set\n", bid, gpu_bufs[bid].host_addr, (void *)gpu_bufs[bid].dma);
        goto set_buf_err;
    }

    *gpu_buf = gpu_bufs[bid].host_addr;     // Set CPU host virtual address
    *(dma_addr_t *)dma = gpu_bufs[bid].dma; // Set physical bus address
    *cuda_addr = gpu_bufs[bid].cuda_addr;       // Set CUDA virtual address (For launching CUDA kernels)
    *page_offset = gpu_bufs[bid].page_offset;   // Set buffer offset within the page

	return 0;

set_buf_err:
	return 1;


}

static int is_gpu_rx_ready(int *num_buf, uint64_t **RX_buffer_pointers, bool (*recycle_grxb_batch)(int batch_ind, int num_req), int (*ixgbe_reg)(int cmd)){

	// Register callbacks immediately, used to interrupt NIC when CUDA buffers are ready.
	ixgbe_callback_recycle_grxb_batch = recycle_grxb_batch;
	ixgbe_callback_register = ixgbe_reg;

	if(gpu_rx_ready && current_RX_buffer_pointers){
		*num_buf = num_gpu_rx_buf;
        *RX_buffer_pointers = (uint64_t *)current_RX_buffer_pointers;

        printk("[GNoM_km] GRX buffers registered. IXGBE Callbacks registered: Recycle: %p, Register: %p\n",
                ixgbe_callback_recycle_grxb_batch,
                ixgbe_callback_register);
	}


	return gpu_rx_ready;
}

static int is_gpu_tx_ready(int *num_buf, uint64_t **TX_buffer_pointers, int (*ixgbe_gnom_tx)(int batch_ind, int buf_ind, unsigned int size)){

    // Register callback to Ixgbe for packet TX
    gnom_tx = ixgbe_gnom_tx;
    
    if(gpu_tx_ready && current_TX_buffer_pointers){
        *num_buf = num_gpu_tx_buf;
        *TX_buffer_pointers = (uint64_t *)current_TX_buffer_pointers;

        printk("[GNoM_km] GTX buffers registered. IXGBE Callback registered: GNoM_Send: %p\n", gnom_tx);
    }
    
    return gpu_tx_ready;
}

static long device_ioctl(struct file *fp, unsigned int cmd, unsigned long arg){

	unsigned long n_cpy = 0;
    int ret = 0;
    int err = 0;
    //unsigned ind=0;
    int count = 0;
    kernel_args_hdr temp_k_hdr;
    gpu_buf_t *tmp_gpu_bufs = NULL;

    nvidia_p2p_page_table_t **tmp_pg_table = NULL;

    unsigned i=0, j=0;
    const char *rx_or_tx;
    const char *rx = "RX";
    const char *tx = "TX";
    int num_gpu_pages = 0;
    int num_gpu_buf = 0;
    int num_gpu_buf_per_page = 0;
    
    // GNOM New
    int num_gpu_bins = 0;


    struct page *tmp_page = NULL;
    void *tmp_page_va = NULL;
    dma_addr_t tmp_dma = 0;

#ifdef DO_GNOM_TX   
    // [0] Index of TX buffer to send
    // [1] Number of requests to send
    // [2] = Sanity check to make sure batch_ind is correct
    int tx_batch_info[3];
    int gnom_km_batch_ind = -1;
#endif

    unsigned long long average_time = 0;

    uint64_t last_dma = 0;
    uint64_t dma_diff = 64*1024;

    uint64_t cuda_addr;
    uint64_t cuda_page_size = 64*1024;

    void *h_ptr_from_cuda_phys = NULL;

    switch(cmd){
        case GPU_REG_SINGLE_BUFFER_CMD:
            // NOTE: This signal is not in use.
            // This signal was used to test a single huge buffer allocation to use GPUDirect 
            // pinning instead of many single pages. This resulted in being able to allocate 
            // significantly more pinned memory than when pinning multiple smaller pages. 

            printk("[GNoM_km]\n\nNOTE: The GPU_REG_SINGLE_BUFFER_CMD signal is a test signal not used for GNoM/MemcachedGPU\n\n");

            // Copy arg header from user
            n_cpy = copy_from_user((void *)&temp_k_hdr,
                                (void __user *)arg,
                                sizeof(kernel_args_hdr));

            if(n_cpy > 0)
                goto hdr_cpy_err;

            printk("[GNoM_km] Copy header success...\n");


            m_kernel_args.num_pages = temp_k_hdr.num_pages;
            m_kernel_args.num_buffers = temp_k_hdr.num_buffers;
            m_kernel_args.buffer_type = temp_k_hdr.buffer_type; // RX or TX

            m_kernel_args.buf_meta_data.gpu_args = (kernel_args *)kmalloc(sizeof(kernel_args), GFP_KERNEL);
            if(!m_kernel_args.buf_meta_data.gpu_args)
                goto malloc_err;

            // Copy over all of the information about GPU buffers from user
            n_cpy = copy_from_user((void *)m_kernel_args.buf_meta_data.gpu_args,
                                (void __user *)temp_k_hdr.buf_meta_data.gpu_args,
                                sizeof(kernel_args));

            if(n_cpy > 0)
                goto buffer_cpy_err;

            printk("[GNoM_km] Copy all CUDA buffer metadata success...\n");

            // Pin and map each buffer
            tmp_pg_table = (nvidia_p2p_page_table_t **)kmalloc(sizeof(nvidia_p2p_page_table_t *), GFP_KERNEL);

            printk("[GNoM_km]: Pinning GPU buffer (%llu MB, %llu B): %p, p2pT: %llu, vaT: %u, page_table: %p\n",
                    m_kernel_args.buf_meta_data.gpu_args->m_size/(1024*1024),
                    m_kernel_args.buf_meta_data.gpu_args->m_size,
                    (void *)m_kernel_args.buf_meta_data.gpu_args->m_addr,
                    m_kernel_args.buf_meta_data.gpu_args->m_tokens.p2pToken,
                    m_kernel_args.buf_meta_data.gpu_args->m_tokens.vaSpaceToken,
                    tmp_pg_table
                    );

            ret = nvidia_p2p_get_pages(
                        m_kernel_args.buf_meta_data.gpu_args->m_tokens.p2pToken,
                        m_kernel_args.buf_meta_data.gpu_args->m_tokens.vaSpaceToken,
                        m_kernel_args.buf_meta_data.gpu_args->m_addr,
                        m_kernel_args.buf_meta_data.gpu_args->m_size,
                        tmp_pg_table,
                        free_callback,
                        tmp_pg_table
                    );

            if(ret || (tmp_pg_table[0]->entries <= 0)){
                printk("[GNoM_km]: ERROR pinning pages :(\n");
            }else{
                printk("[GNoM_km]: SUCCESSFULLY PINNED PAGES!! CHECK NVIDIA-SMI BAR USAGE!: # of entries: %u \n", tmp_pg_table[0]->entries);
            }

            last_dma = tmp_pg_table[0]->pages[0]->physical_address;
            for(i=1; i<tmp_pg_table[0]->entries; ++i){
                if((tmp_pg_table[0]->pages[i]->physical_address - last_dma) != dma_diff){
                    printk("[GNoM_km]: ERROR DMA ADDRESS NOT CONTIGUOUS :( :( \n");
                    break;
                }
                last_dma = tmp_pg_table[0]->pages[i]->physical_address;
            }

            if(m_kernel_args.buf_meta_data.gpu_args)
                kfree(m_kernel_args.buf_meta_data.gpu_args);

            break;

        case GPU_REG_MULT_BUFFER_CMD: // User CUDA application registers multiple GPU buffers to NIC driver using GPUDirect
            // This signal is called with a single large buffer allocation, which is split into multiple smaller
            // buffers internally. This enables the maximum amount of pinned GPUDirect memory. Previously tried to 
            // pin individual 2KB buffers with GPUDirect but resulted in significantly less pinnable memory.

        	// Copy arg header from user
        	n_cpy = copy_from_user((void *)&temp_k_hdr,
								(void __user *)arg,
								sizeof(kernel_args_hdr));

        	if(n_cpy > 0)
        		goto hdr_cpy_err;

        	printk("[GNoM_km] Copy header success...\n");

			m_kernel_args.num_pages = temp_k_hdr.num_pages;
        	m_kernel_args.num_buffers = temp_k_hdr.num_buffers;
            m_kernel_args.buffer_type = temp_k_hdr.buffer_type; // RX or TX


            m_kernel_args.buf_meta_data.gpu_args = (kernel_args *)kmalloc(sizeof(kernel_args), GFP_KERNEL);
            if(!m_kernel_args.buf_meta_data.gpu_args)
                goto malloc_err;

            // Copy over all of the information about GPU buffers from user
            n_cpy = copy_from_user((void *)m_kernel_args.buf_meta_data.gpu_args,
                                (void __user *)temp_k_hdr.buf_meta_data.gpu_args,
                                sizeof(kernel_args));

            if(n_cpy > 0)
                goto buffer_cpy_err;


			printk("[GNoM_km] Copy all CUDA buffer metadata success...\n");

			/* Calculate the number of pages, buffers, and buffers per page */
			num_gpu_pages = m_kernel_args.num_pages;
			num_gpu_buf = m_kernel_args.num_buffers;
			num_gpu_buf_per_page = num_gpu_buf / num_gpu_pages;


			if(num_gpu_buf_per_page*num_gpu_pages != num_gpu_buf)
				goto buffer_cpy_err;


            tmp_gpu_bufs = vmalloc(num_gpu_buf*sizeof(gpu_buf_t)); // Use VMALLOC for large # of buffers
            if(!tmp_gpu_bufs)
                goto malloc_err;

			printk("[GNoM_km] Pinning and registering %d pages...\n", num_gpu_pages);


			tmp_pg_table = (nvidia_p2p_page_table_t **)kmalloc(sizeof(nvidia_p2p_page_table_t *), GFP_KERNEL);
            if(!tmp_pg_table)
			    goto malloc_err;


            printk("[GNoM_km]: Pinning GPU buffer (%llu MB, %llu B): %p, p2pT: %llu, vaT: %u, page_table_ptr: %p\n",
                    m_kernel_args.buf_meta_data.gpu_args->m_size/(1024*1024),
                    m_kernel_args.buf_meta_data.gpu_args->m_size,
                    (void *)m_kernel_args.buf_meta_data.gpu_args->m_addr,
                    m_kernel_args.buf_meta_data.gpu_args->m_tokens.p2pToken,
                    m_kernel_args.buf_meta_data.gpu_args->m_tokens.vaSpaceToken,
                    tmp_pg_table
                    );

            // Only do one large buffer map. nvidia_pg_table contains all pages in single call
            ret = nvidia_p2p_get_pages(
                        m_kernel_args.buf_meta_data.gpu_args->m_tokens.p2pToken,
                        m_kernel_args.buf_meta_data.gpu_args->m_tokens.vaSpaceToken,
                        m_kernel_args.buf_meta_data.gpu_args->m_addr,
                        m_kernel_args.buf_meta_data.gpu_args->m_size,
                        tmp_pg_table,
                        free_callback,
                        tmp_pg_table
                    );

            if(ret || (tmp_pg_table[0]->entries <= 0) || (num_gpu_pages != tmp_pg_table[0]->entries)){
                printk("[GNoM_km]: ERROR pinning pages :( (%d, %u)\n", num_gpu_pages, tmp_pg_table[0]->entries);
                goto p2p_get_pages_err;
            }else{
                printk("[GNoM_km]: SUCCESSFULLY PINNED %u GPU PAGES!\n", tmp_pg_table[0]->entries);
            }

            last_dma = tmp_pg_table[0]->pages[0]->physical_address;
            for(i=1; i<tmp_pg_table[0]->entries; ++i){
                if((tmp_pg_table[0]->pages[i]->physical_address - last_dma) != dma_diff){
                    printk("[GNoM_km]: ERROR DMA ADDRESS NOT CONTIGUOUS :( :( \n");
                    break;
                }
                last_dma = tmp_pg_table[0]->pages[i]->physical_address;
            }

            // Now setup all internal GNoM buffer structures
            count = 0;
            cuda_addr = m_kernel_args.buf_meta_data.gpu_args->m_addr;
            for(i=0; i<num_gpu_pages; ++i){
                count++;
                h_ptr_from_cuda_phys = (void *)ioremap(tmp_pg_table[0]->pages[i]->physical_address, cuda_page_size);

                for(j=0; j<num_gpu_buf_per_page; ++j){
                    tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].cuda_addr    = cuda_addr; // Set to VA of each CUDA page (Calculated by offsetting original VA)
                    tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].host_addr    = h_ptr_from_cuda_phys; // Map virtual address
                    tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].user_pg      = NULL; // No CPU page for GPU buffer
                    tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].dma          = tmp_pg_table[0]->pages[i]->physical_address;               // DMA address, physical bus address
                    tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].page_offset  = j*RX_BUFFER_SZ; // Offset of this buffer within the physical page
                }

                cuda_addr += cuda_page_size;
            }

			if(m_kernel_args.buf_meta_data.gpu_args)
			    kfree(m_kernel_args.buf_meta_data.gpu_args);


            // Ensure all buffers are set before enabling flag to NIC
            rmb();
            
            if (m_kernel_args.buffer_type == GNOM_RX) {
                // RX buffer mapping
                m_gpu_rx_bufs = tmp_gpu_bufs;
                page_table_rx = tmp_pg_table; 
                num_gpu_rx_pages = num_gpu_pages;
                num_gpu_rx_buf = num_gpu_buf;
                num_gpu_rx_buf_per_page = num_gpu_buf_per_page;
                rx_or_tx = rx;
       			gpu_rx_ready = 1;
            }else{
                // TX buffer mapping
                m_gpu_tx_bufs = tmp_gpu_bufs;
                page_table_tx = tmp_pg_table;
                num_gpu_tx_pages = num_gpu_pages;
                num_gpu_tx_buf = num_gpu_buf;
                num_gpu_tx_buf_per_page = num_gpu_buf_per_page;
                rx_or_tx = tx;
                gpu_tx_ready = 1;
            }

			printk("[GNoM_km]: %s CUDA buffers successfully registered with GPU_km...\n", rx_or_tx);
			printk("[GNoM_km]: \t Number of pages:\t\t %d\n", num_gpu_pages);
			printk("[GNoM_km]: \t Number of buffers:\t\t %d\n", num_gpu_buf);
			printk("[GNoM_km]: \t Number of bins:\t\t %d\n", num_gpu_bins);
			printk("[GNoM_km]: \t Number of buffers per page:\t %d\n", num_gpu_buf_per_page);

			if(!ixgbe_callback_register)
				goto p2p_get_pages_err; // Free memory and return

			printk("[GNoM_km]: Registration successful, call IOCTL with SIGNAL_NIC cmd to initialize the NIC\n");
        	break;

        case SIGNAL_NIC:
			printk("[GNoM_km]: Calling to restart NIC\n");
			ixgbe_callback_register(0); // Signal/reset NIC now that GPU buffers are registered.
        	break;

        case STOP_SYSTEM:
            printk("[GNoM_km]: STOP_SYSTEM signal received\n");
            IS_DONE_FLAG = 1;
            gpu_rx_ready = 0;
            gpu_tx_ready = 0;

            // Print GNoM_ixgbe stats
            ixgbe_callback_register(4);

            // Print this stats
            if(total_count > 0){
                average_time = total_time / total_count;
                printk("[GNOM_km]: Complete receive to send total time for %lld buffers: %lld - average = %lld ns (%lld us)\n", total_time, total_count, average_time, average_time/1000);
            }

            pending_batch_tail++;
            rmb();
            wake_up(&m_req_wq); // Wake up any potentially waiting 
            ixgbe_callback_register(0);
            break;

        case SHUTDOWN_NIC:
            if(!IS_DONE_FLAG){
                printk("[GNoM_km]: Calling to shutdown NIC\n");
                gpu_rx_ready = 0; // Return NIC to original CPU only mode
                gpu_tx_ready = 0;
                ixgbe_callback_register(0); // Signal/reset NIC now that GPU buffers are registered.
            }
        	break;

        case GNOM_TX_SEND:
            // Copy arg header from user
#ifdef DO_GNOM_TX
            n_cpy = copy_from_user((void *)tx_batch_info,
                                (void __user *)arg,
                                3*sizeof(int));

            if(unlikely((n_cpy > 0) || !gnom_tx || m_lw_pending_batches[tx_batch_info[0]].batch_id != tx_batch_info[2]))
                goto hdr_cpy_err;

            gnom_km_batch_ind = m_lw_pending_batches[tx_batch_info[0]].tx_batch_ind;
            // Now call gnom_tx for every request in the batch
            for(i=0; i<tx_batch_info[1]; ++i){
                gnom_tx(gnom_km_batch_ind, i, 2048 /* FIXME */);
            }
#else
            printk("[GNoM_km]: Error - GNoM is not configured to run TX through GPUDirect. Please define DO_GNOM_TX\n");
#endif
            break;


        case GNOM_REG_MULT_CPU_BUFFERS:
            // This signal is used for GNoM TX. DO_GNOM_TX should be set
            // User CUDA application registers multiple GPU-accessible CPU buffers to NIC driver
            gnom_print("Registering GPU-accessible CPU buffers\n");
            
#ifndef DO_GNOM_TX
            printk("[GNoM_km]: Error - GNoM is not configured to run TX. Please define DO_GNOM_TX\n");
#endif

            if(gnom_dev == NULL)
                goto dev_not_set_err;
            
            // Copy arg header from user
            n_cpy = copy_from_user((void *)&temp_k_hdr,
                                   (void __user *)arg,
                                   sizeof(kernel_args_hdr));
            
            if(n_cpy > 0)
                goto hdr_cpy_err;

            m_kernel_args.num_pages = temp_k_hdr.num_pages;
            m_kernel_args.num_buffers = temp_k_hdr.num_buffers;
            m_kernel_args.buffer_type = temp_k_hdr.buffer_type; // RX or TX
            
            m_kernel_args.buf_meta_data.cpu_args = (cpu_kernel_args *)kmalloc(m_kernel_args.num_pages*sizeof(cpu_kernel_args), GFP_KERNEL);
            if(!m_kernel_args.buf_meta_data.cpu_args)
                goto malloc_err;
            
            // Copy over all of the information about GPU buffers from user
            n_cpy = copy_from_user((void *)m_kernel_args.buf_meta_data.cpu_args,
                                   (void __user *)temp_k_hdr.buf_meta_data.cpu_args,
                                   sizeof(cpu_kernel_args)*m_kernel_args.num_pages);
            
            if(n_cpy > 0)
                goto buffer_cpy_err;

            /* Calculate the number of pages, buffers, and buffers per page */
            num_gpu_pages = m_kernel_args.num_pages;
            num_gpu_buf = m_kernel_args.num_buffers;
            num_gpu_buf_per_page = num_gpu_buf / num_gpu_pages;
            

            if(num_gpu_buf_per_page*num_gpu_pages != num_gpu_buf)
                goto buffer_cpy_err;
            

            tmp_gpu_bufs = kmalloc(num_gpu_buf*sizeof(gpu_buf_t), GFP_KERNEL);
            if(!tmp_gpu_bufs)
                goto malloc_err;
            
            printk("[GNoM_km] Pinning and registering %d GPU-accessible CPU pages, %d buffers, in %d bins\n", num_gpu_pages, num_gpu_buf, num_gpu_bins);
            
            down_read(&current->mm->mmap_sem);
            for(i=0; i<num_gpu_pages; ++i){
                count++;
                // (1) Get the user page
                // (2) Map the user page
                // (3) Setup DMA mapping
                // (4) Setup GNoM data structures

                // (1)
                err = get_user_pages(current,
                                     current->mm,
                                     (unsigned long)m_kernel_args.buf_meta_data.cpu_args[i].user_page_va,
                                     1,
                                     1,
                                     1,
                                     &tmp_page,
                                     NULL);
                
                if(err == 1){
                    
                    // (2)
                    tmp_page_va = kmap(tmp_page);
                    
                    // (3)
                    tmp_dma = dma_map_page(gnom_dev, tmp_page, 0, CPU_PAGE_SIZE, DMA_TO_DEVICE);
                    
                    if (dma_mapping_error(gnom_dev, tmp_dma)){
                        up_read(&current->mm->mmap_sem);
                        goto dma_err;
                    }
                    
                    if(!tmp_dma || !tmp_page){
                        up_read(&current->mm->mmap_sem);
                        goto dma_err;
                    }

                    // Make sure everything is synced for the CPU/GPU to access
                    dma_sync_single_range_for_cpu(gnom_dev,
                                                  tmp_dma,
                                                  0,
                                                  CPU_PAGE_SIZE,
                                                  DMA_TO_DEVICE);
                    

                    // (4)
                    for(j=0; j<num_gpu_buf_per_page; ++j){
                        tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].cuda_addr = m_kernel_args.buf_meta_data.cpu_args[i].cuda_page_va;
                        tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].host_addr = tmp_page_va;
                        tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].user_pg = tmp_page;
                        tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].dma = tmp_dma;               // DMA address, physical bus address
                        tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)].page_offset = j*TX_BUFFER_SZ; // Offset of this buffer within the physical page
                        // print_gpu_buf_t(&tmp_gpu_bufs[j + (i*num_gpu_buf_per_page)], j + (i*num_gpu_buf_per_page));
                    }
                }else{
                    printk("[GNoM_km]: Error with get_user_pages on a mult page mapping (%d)\n", i);
                }
            }
            up_read(&current->mm->mmap_sem);
            
            // Free up the cpu_args structure
            kfree(m_kernel_args.buf_meta_data.cpu_args);
            
            // Ensure all buffers are set before enabling flag to NIC
            rmb();
            
            if (m_kernel_args.buffer_type == GNOM_RX) {
                // RX buffer mapping

                m_gpu_rx_bufs = tmp_gpu_bufs;
                page_table_rx = NULL; // No CUDA page table for CPU resident buffers
                num_gpu_rx_pages = num_gpu_pages;
                num_gpu_rx_buf = num_gpu_buf;
                num_gpu_rx_buf_per_page = num_gpu_buf_per_page;
                num_gpu_rx_bins = num_gpu_bins;
                rx_or_tx = rx;
                gpu_rx_ready = 1;
            }else{
                // TX buffer mapping
                m_gpu_tx_bufs = tmp_gpu_bufs;
                page_table_tx = NULL; // No CUDA page table for CPU resident buffers
                num_gpu_tx_pages = num_gpu_pages;
                num_gpu_tx_buf = num_gpu_buf;
                num_gpu_tx_buf_per_page = num_gpu_buf_per_page;
                num_gpu_tx_bins = num_gpu_bins;
                rx_or_tx = tx;
                gpu_tx_ready = 1;
            }
            
            printk("[GNoM_km]: %s GPU-accessible CPU CUDA buffers successfully registered with GPU_km\n", rx_or_tx);
            printk("[GNoM_km]: \t Number of pages:\t\t %d\n", num_gpu_pages);
            printk("[GNoM_km]: \t Number of buffers:\t\t %d\n", num_gpu_buf);
            printk("[GNoM_km]: \t Number of bins:\t\t %d\n", num_gpu_bins);
            printk("[GNoM_km]: \t Number of buffers per page:\t %d\n", num_gpu_buf_per_page);
            
            break;
        
        case GNOM_UNREG_MULT_CPU_BUFFERS:

            if(gnom_dev == NULL)
                goto dev_not_set_err;

            // Copy arg header from user
            n_cpy = copy_from_user((void *)&temp_k_hdr,
                                   (void __user *)arg,
                                   sizeof(kernel_args_hdr));

            if(n_cpy > 0)
                goto hdr_cpy_err;

            if(temp_k_hdr.buffer_type == GNOM_RX){
                gnom_print("Unregistering GRXBs\n");
                tmp_gpu_bufs = m_gpu_rx_bufs;
                num_gpu_buf = num_gpu_rx_buf;
                num_gpu_pages = num_gpu_rx_pages;
                num_gpu_buf_per_page = num_gpu_rx_buf_per_page;
                num_gpu_bins = num_gpu_rx_bins;
            }else{
                gnom_print("Unregistering GTXBs\n");
                tmp_gpu_bufs = m_gpu_tx_bufs;
                num_gpu_buf = num_gpu_tx_buf;
                num_gpu_pages = num_gpu_tx_pages;
                num_gpu_buf_per_page = num_gpu_tx_buf_per_page;
                num_gpu_bins = num_gpu_tx_bins;
            }

            printk("[GNoM_km]: \t Number of pages:\t\t %d\n", num_gpu_pages);
            printk("[GNoM_km]: \t Number of buffers:\t\t %d\n", num_gpu_buf);
            printk("[GNoM_km]: \t Number of bins:\t\t %d\n", num_gpu_bins);
            printk("[GNoM_km]: \t Number of buffers per page:\t %d\n", num_gpu_buf_per_page);
            count=0;

            down_read(&current->mm->mmap_sem);
            for(i=0; i<num_gpu_pages; i++){
                if(tmp_gpu_bufs[i*num_gpu_buf_per_page].user_pg != NULL){ // First buffer in the page will clean things up, then null out everything else
                    count++;

                    // Tear down DMA mapping
                    dma_unmap_page(gnom_dev,
                            tmp_gpu_bufs[i*num_gpu_buf_per_page].dma,
                                   CPU_PAGE_SIZE,
                                   DMA_TO_DEVICE);

                    // Unmap page
                    kunmap(tmp_gpu_bufs[i*num_gpu_buf_per_page].user_pg);

                    // Set dirty bit on page and release the page cache
                    if(!PageReserved(tmp_gpu_bufs[i*num_gpu_buf_per_page].user_pg))
                        SetPageDirty(tmp_gpu_bufs[i*num_gpu_buf_per_page].user_pg);
                    page_cache_release(tmp_gpu_bufs[i*num_gpu_buf_per_page].user_pg);

                    for(j=0; j<num_gpu_buf_per_page; ++j){
                        // Clear the buffer info
                        tmp_gpu_bufs[j + i*num_gpu_buf_per_page].user_pg = NULL;
                        tmp_gpu_bufs[j + i*num_gpu_buf_per_page].host_addr = NULL;
                        tmp_gpu_bufs[j + i*num_gpu_buf_per_page].cuda_addr = 0;
                        tmp_gpu_bufs[j + i*num_gpu_buf_per_page].dma = 0;
                    }


                }
            }

            up_read(&current->mm->mmap_sem);
            
            printk("[GNoM_km]: Successfully unregistered %d GPU accessible CPU pages\n", count);
            break;

        case TEST_SEND_SINGLE_PACKET:
            printk("[GNoM_km]: Sending packet\n");
            ixgbe_callback_register(3);
            printk("[GNoM_km]: Sending packet complete...\n");

            break;

        case TEST_CHECK_SEND_COMPLETE:
            break;

        default:
        	goto ioctl_err;
    }

    return 0;

// TODO: Make sure all error paths below correctly clean up any partially allocated structures.
dma_err:
    printk("Err: [GNoM_km] DMA mapping failed\n");
    return -EINVAL;
    
dev_not_set_err:
    printk("Err: [GNoM_km] Device not set yet\n");
    return -EINVAL;
hdr_cpy_err:
	printk("Err: [GNoM_km] Asked to copy %lu bytes, only copied %lu bytes\n", sizeof(kernel_args), sizeof(kernel_args)-n_cpy);
	return -EINVAL;
buffer_cpy_err:
	printk("Err: [GNoM_km] Asked to copy %lu bytes, only copied %lu bytes\n", sizeof(kernel_args)*m_kernel_args.num_buffers, sizeof(kernel_args)*m_kernel_args.num_buffers-n_cpy);
	return -EINVAL;
malloc_err:
	printk("[GNoM_km] Failed to allocate kernel args buffer...\n");
	return -EINVAL;
p2p_get_pages_err:

    if(tmp_gpu_bufs)
        vfree(tmp_gpu_bufs);
    if(m_kernel_args.buf_meta_data.gpu_args)
        kfree(m_kernel_args.buf_meta_data.gpu_args);

	printk("\t[GNoM_km] Failed to pin the GPU buffer (%d) :( (%d successfully registered)\n", ret, count);
	return -EINVAL;
ioctl_err:
	pr_err("Invalid cmd: %u\n", cmd);
	return -EINVAL;

}

// Sets ixgbe struct device * in GNoM - used for setting up TX DMA
int set_device(struct device *dev){
    if(gnom_dev){ // Device already set
       	printk("[GNoM_km]: Device already set (dev=%p)\n", gnom_dev);
        return -1;
    }else{
        gnom_dev = dev;
        printk("[GNoM_km]: Device set (dev=%p)\n", gnom_dev);
        return 0;
    }
}

int get_dma_info(struct page **kernel_page, void **kernel_va_addr, int set_finished){
    if(!set_finished){
        *kernel_page = su_page;
        *kernel_va_addr = sk_address;
    }else{
        is_send_finished = 1;
    }
    return 0;
}

// Callback function for NVIDIA GPUDirect. Registered when pinning the GRXBs, called when freeing the GRXBs.
void free_callback(void *data){
	struct nvidia_p2p_page_table *page_table = *((struct nvidia_p2p_page_table **)data);

	printk("[GNoM_km] Free_callback... page_table: %p\n", page_table);
	if(page_table)
		nvidia_p2p_free_page_table(page_table);

}


static long num_ready = 0;
static long num_wait = 0;

static ssize_t device_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos){

	int m_queue_ind = 0;
	int num_req =0;
	int n_cpy = 0;
	int try_count = 0;
	long timeout_ret = 0;

	// Note: Now locking in GNoM-User - Obtain lock
	//if(down_interruptible(&nic_gpu_lock))
	//	return -ERESTARTSYS;

	if(unlikely(pending_batch_head == pending_batch_tail)){ // Empty
		num_host_thread_waits++;
		// No work, need to wait
#ifdef DEBUG_PRINT
		printk("[GNoM_km] PID: %d waiting for work!\n", current->pid);
#endif
		while(pending_batch_head == pending_batch_tail){
			//up(&nic_gpu_lock); // Note: Now locking in GNoM-User - Release lock
			timeout_ret = wait_event_timeout(m_req_wq, (pending_batch_tail != pending_batch_head), 2500); // Go to sleep
#ifdef DEBUG_PRINT            
			printk("[GNoM_km] Thread %d alive and trying the condition again (%d/10)...\n", current->pid, try_count);
#endif            
			if(timeout_ret == 0)
				try_count++;

			if(IS_DONE_FLAG || (try_count > 20)){
			    // Either host process has signaled us to complete or we've waited a LONG time for work that will never come. Bail out. 
			    goto read_err;
			}

			//if(down_interruptible(&nic_gpu_lock)) // Note: Now locking in GNoM-User - Obtain the lock again, only a single waiting thread will make it passed here
			//	return -ERESTARTSYS;

		}
        num_wait++;
	}else{
        num_ready++;
    }

#ifdef DEBUG_PRINT
	printk("[GNoM_km]: Dispatching buffers to GNoM_pre\n");
#endif

	// Have our element, increment the tail and release the lock
	m_queue_ind = pending_batch_tail;
	pending_batch_tail = (pending_batch_tail+1) % MAX_BATCH_QUEUE; // Update the tail pointer
	num_req = m_lw_pending_batches[m_queue_ind].num_reqs;
    
    //up(&nic_gpu_lock); // Note: Now locking in GNoM-User - Release lock, we've already got the right index and moved the tail pointer up

	n_cpy = copy_to_user(buf, &m_lw_pending_batches[m_queue_ind].rx_batch_ind, sizeof(int));
    buf += sizeof(int);
    n_cpy = copy_to_user(buf, &m_lw_pending_batches[m_queue_ind].tx_batch_ind, sizeof(int));
    buf += sizeof(int);
    n_cpy += copy_to_user(buf, &m_lw_pending_batches[m_queue_ind].batch_id, sizeof(int));
    buf += sizeof(int);
    n_cpy += copy_to_user(buf, &m_queue_ind, sizeof(int)); 

    if(unlikely(n_cpy > 0)){
		printk("[GNoM_km] Error: Did not copy enough data on device_read...(%d / %lu)\n", (int)(4*sizeof(int) - n_cpy), 4*sizeof(int));
		goto read_err;
	}

	return num_req; // Return number of requests read

read_err:
	printk("[GNoM_km] Error: Worker-thread %d bailing out...\n", current->pid);
	return 0; // Error, no data written
}

/*
 * This function will be used to return the RXB to the NIC's freelist
 */

static ssize_t device_write(struct file *filp, const char __user *buf, size_t count, loff_t * ppos){

    // First - Recycle GRXB to GNoM_ixgbe
    // Second - Perform TX
    // FIXME: TX was only part of testing - this should be moved to a separate IOCTL call

	int v_batch_id = 0;
	int batch_ind = 0;
	int n_cpy;
	const char __user *t_buf = buf;

#ifdef DO_GNOM_TX
	int i=0;
#endif

	n_cpy = copy_from_user((void *)&v_batch_id, t_buf, sizeof(int));
	t_buf += sizeof(int);
	n_cpy += copy_from_user((void *)&batch_ind, t_buf, sizeof(int));

	if(unlikely((n_cpy > 0) || !ixgbe_callback_recycle_grxb_batch || m_lw_pending_batches[batch_ind].batch_id != v_batch_id))
		goto write_err;

#ifdef DEBUG_PRINT
	printk("[GNoM_km]: Recycling buffers (batch_ind = %d)\n", batch_ind);
#endif

	// Return the used GRXBs to the NIC's freelist
	ixgbe_callback_recycle_grxb_batch(m_lw_pending_batches[batch_ind].rx_batch_ind, MAX_REQ_BATCH_SIZE);

#ifdef DO_GNOM_TX
#ifdef DEBUG_PRINT    
	//printk("[GNoM_km]: Testing packet send\n");
#endif
	for(i=0; i<MAX_REQ_BATCH_SIZE; ++i){
	    gnom_tx(m_lw_pending_batches[batch_ind].tx_batch_ind, i, 1024);
	}
#endif

    return 0;

write_err:
    printk("[GNoM_km]: Error on buffer recycle... (n_cpy=%d, call_back=%p, batch_id=%d, v_batch_id: %d)\n",
            n_cpy, ixgbe_callback_recycle_grxb_batch, m_lw_pending_batches[batch_ind].batch_id, v_batch_id);
	return 1;
}

// Structure and code borrowed from NTOP PF_RING to set hooks for GNoM_ND (Modified Intel IXGBE network driver).
static int gpu_notifier(struct notifier_block *this, unsigned long msg, void *data){
	struct net_device *dev = netdev_notifier_info_to_dev(data);
	ubc_gpu_hooks *hook;

	if(dev != NULL) {
		/* Skip non ethernet interfaces */
		if(	(dev->type != ARPHRD_ETHER)
			&& (dev->type != ARPHRD_IEEE80211)
			&& (dev->type != ARPHRD_IEEE80211_PRISM)
			&& (dev->type != ARPHRD_IEEE80211_RADIOTAP)
			&& strncmp(dev->name, "bond", 4)) {
				return NOTIFY_DONE;
		}

		switch(msg) {
			case NETDEV_PRE_UP:
			case NETDEV_UP:
			case NETDEV_DOWN:
				break;
			case NETDEV_REGISTER:
				printk("\t[GNoM_km] packet_notifier(%s) [REGISTER][hook_to_gpu_mod=%p][hook=%p]: Setting GNoM-ND hook in GNoM-KM\n", 
                        dev->name, dev->hook_to_gpu_mod, &gpu_hooks);

                // Don't worry about overwriting any previous hooks, if any. 
    			dev->hook_to_gpu_mod = &gpu_hooks;

				break;

			case NETDEV_UNREGISTER:
                // TODO: Add any necessary cleanup here. Currently just unsetting the hook.
				printk("\t[GNoM_km] packet_notifier(%s) [UNREGISTER][hook_to_gpu_mod_ptr=%p]. Unregistering the GNoM-ND hook.\n",
				        dev->name, dev->hook_to_gpu_mod);
				hook = (ubc_gpu_hooks*)dev->hook_to_gpu_mod;
				if(hook) { 
					dev->hook_to_gpu_mod = NULL;
				}
			break;

			case NETDEV_CHANGE:     
			case NETDEV_CHANGEADDR: 
			case NETDEV_CHANGENAME:
				break;
			default:
				printk("[GNoM_km]: packet_notifier(%s): No handler for signal [msg=%lu][hook_to_gpu_mod=%p]\n", dev->name, msg, dev->hook_to_gpu_mod);
				break;
		}
	}

	return NOTIFY_DONE;
}


static int __init gpu_km_init(void){
	int ret=0;
	
	printk("\n\n[GNoM_km] Initializing GNoM-KM...\n\n");  

	IS_DONE_FLAG = 0;

	pending_batch_head = 0;
	pending_batch_tail = 0;
	pkt_rcv_cnt = 0;
	num_gpu_rx_buf = 0;
	num_gpu_tx_buf = 0;

	gpu_rx_ready = 0;
	gpu_tx_ready = 0;

	sema_init(&nic_gpu_lock, 1);
	sema_init(&user_gpu_lock, 1);

	m_lw_pending_batches = (lw_network_requests *)kmalloc( sizeof(lw_network_requests)*MAX_BATCH_QUEUE, GFP_KERNEL);

	/* Register netdevice notifer */
	ret = register_netdevice_notifier(&gpu_netdev_notifier);

    /* Register the character device */
    ret = register_chrdev (GPU_KM_MAJOR, DEVICE_NAME, &gpu_km_ops);
	if (ret < 0) {
		printk("gpu_km_init: failed with %d\n", ret);
		return ret;
	}

	gpu_km_class = class_create (THIS_MODULE, "gpu_km");
    if(gpu_km_class == NULL){
        printk("gpu_km_init: failed class_create\n");
    }
    
	device_create (gpu_km_class, NULL, MKDEV (GPU_KM_MAJOR, 0), NULL, "gpu_km");

	return 0;
}

static void __exit gpu_km_exit(void){
	printk("\n\n[GNoM_km]: Exitting GNoM-KM...\n\n");

    IS_DONE_FLAG = 1;

	if(m_lw_pending_batches)
	    kfree(m_lw_pending_batches);

	if(m_gpu_rx_bufs){
	    vfree(m_gpu_rx_bufs);
	}

    if(m_gpu_tx_bufs){
        vfree(m_gpu_tx_bufs);
    }

	if(page_table_rx)
		kfree(page_table_rx);

    if(page_table_tx)
        kfree(page_table_tx);
	
    device_destroy (gpu_km_class, MKDEV (GPU_KM_MAJOR, 0));
	unregister_netdevice_notifier(&gpu_netdev_notifier);
	class_destroy (gpu_km_class);
	unregister_chrdev (GPU_KM_MAJOR, DEVICE_NAME);
}

module_init(gpu_km_init);
module_exit(gpu_km_exit);

MODULE_LICENSE ("Dual BSD/GPL");
MODULE_AUTHOR ("Tayler Hetherington");
MODULE_DESCRIPTION ("GNoM: GPU Network Offload Manager");

