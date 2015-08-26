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
 * cuda_memory_manager.cpp
 */

// Allocate GPU hash table and GPU lock table memory.

extern "C" {
#include "cuda_memory_manager.h"
}

// CUDA utilities and system includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Data for memory buffers and meta data for memory manager
static mem_data m_mem_data;

// Ensure memory is initialized before mem_alloc or mem_free is called
static bool mem_inited = false;

void gpu_mem_init(CUDAContext *cuda_context, int hashtable_size, int hashlock_size){
	CUresult status;
	assert((hashtable_size > 0) && (hashlock_size));

    status = cuCtxPushCurrent(cuda_context->hcuContext);
    if(status != CUDA_SUCCESS){
        printf("Context push error: %d\n", status);
        exit(1);
    }

	int *htb = new int[hashtable_size/sizeof(int)];
	int *hlb = new int[hashlock_size/sizeof(int)];
	memset(htb, 0, hashtable_size);
	memset(hlb, 0, hashlock_size);

	// Allocate buffers
	printf("Allocating hashtable of %.2lf MB, %lu entries...\n", hashtable_size / (1024.0 * 1024.0), hashtable_size / sizeof(gpu_primary_hashtable));
    status = cuMemAlloc( &m_mem_data.d_gpu_hashtable_ptr, hashtable_size );
    if(status != CUDA_SUCCESS){
        printf("Hashtable/lock memory allocation/copy failed: %d... \n", status);
        abort();
    }

    printf("Allocating hash lock of %.2lf MB, %lu entries...\n", hashlock_size / (1024.0 * 1024.0), hashlock_size / sizeof(gpu_primary_hashtable));
    status = cuMemAlloc( &m_mem_data.d_gpu_lock_ptr, hashlock_size );
    if(status != CUDA_SUCCESS){
        printf("Hashtable/lock memory allocation/copy failed: %d... \n", status);
        abort();
    }

    // Clear buffers to 0
    status = cuMemcpyHtoD(m_mem_data.d_gpu_hashtable_ptr, htb, hashtable_size);
    if(status != CUDA_SUCCESS){
        printf("Hashtable/lock memory allocation/copy failed: %d... \n", status);
        abort();
    }

    status = cuMemcpyHtoD(m_mem_data.d_gpu_lock_ptr, hlb, hashlock_size);
    if(status != CUDA_SUCCESS){
        printf("Hashtable/lock memory allocation/copy failed: %d... \n", status);
        abort();
    }

    // Cleanup
    delete htb;
    delete hlb;

    cuCtxPopCurrent(NULL);

    mem_inited = true;

}

void gpu_mem_close(){

    CUresult status;

    if(m_mem_data.d_gpu_hashtable_ptr){
        status = cuMemFree(m_mem_data.d_gpu_hashtable_ptr);
    }

    if(m_mem_data.d_gpu_lock_ptr){
        status = cuMemFree(m_mem_data.d_gpu_lock_ptr);
    }
}


// Allocate Primary Hashtable on GPU (Stores full key + pointer to the item on CPU memory)
CUdeviceptr get_GPU_hashtable(){
    return m_mem_data.d_gpu_hashtable_ptr;
}

CUdeviceptr get_GPU_lock(){
	return m_mem_data.d_gpu_lock_ptr;
}
