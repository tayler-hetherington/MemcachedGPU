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
 * cuda_stream_wrapper.c
 */

extern "C" {
#include "cuda_stream_wrapper.h"
}

#include "cuda_stream_manager.h"

static cuda_stream_manager_base *m_csm = NULL;

// test_to_run
//  0: Main on-line GPUDirect configuration
//  1: On-line non-GPUDirect configuration
void init_stream_manager_gpu_NoM(CUDAContext *context, CUdeviceptr gpu_hashtable_ptr, CUdeviceptr gpu_lock_ptr, int test_to_run){
	if(m_csm == NULL){
	    switch(test_to_run){
	    case 0:
	        m_csm = new cuda_stream_manager(context, gpu_hashtable_ptr, gpu_lock_ptr);
	        break;
	    case 1:
	        m_csm = new cuda_stream_manager_ngd(context, gpu_hashtable_ptr, gpu_lock_ptr);
	        break;
	    default: // Default to the regular GNoM GPUDirect framework
	        m_csm = new cuda_stream_manager(context, gpu_hashtable_ptr, gpu_lock_ptr);
	        break;
	    }

	    assert(m_csm);
	}else{
	    printf("Error: Trying to re-initialize GNoM. Aborting...\n");
	    abort();
	}

	if(m_csm){
		printf("CUDA Stream Manager Setup Complete...\n");
	}else{
		printf("**ERROR** Could not initialize the Stream Manager... Aborting\n");
		abort();
	}
}

int send_set_stream_request(void *req, int req_size, void *res, int res_size, rel_time_t timestamp){
	assert(m_csm);
	return m_csm->set_update_gpu_hashtable(req, req_size, res, res_size, timestamp);
}

int gnom_set_lock(){
    assert(m_csm);
    return m_csm->set_lock();
}

int gnom_set_unlock(){
    assert(m_csm);
    return m_csm->set_unlock();
}

int gnom_poll_get_complete_timestamp(int set_evict_timestamp){
    assert(m_csm);
    return m_csm->poll_get_complete_timestamp(set_evict_timestamp);
}

void inc_set_hit(){
    assert(m_csm);
    m_csm->inc_set_hit();
}

void inc_set_miss(){
    assert(m_csm);
    m_csm->inc_set_miss();
}

void inc_set_evict(){
    assert(m_csm);
    m_csm->inc_set_evict();
}

void stop_gpu_stream(){
    assert(m_csm);
    //m_csm->stop();

    /*** GPU-NoM ***/
    m_csm->stop_gpu_nom();
}





