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
 * gpu_km_shared.h
 */

#ifndef GPU_KM_SHARED_H_
#define GPU_KM_SHARED_H_


#define NUM_REQUESTS_PER_BATCH 512

#define MAX_REQ_BATCH_SIZE  NUM_REQUESTS_PER_BATCH // TODO: Just replace all MAX_REQ_BATCH_SIZE
#define MAX_BATCH_QUEUE	512 //256 // Max number of batches able to be queued in GNoM_km, each containing <= MAX_REQ_BATCH_SIZE


#define GPU_REG_SINGLE_BUFFER_CMD	96
#define GPU_REG_MULT_BUFFER_CMD	97
#define SIGNAL_NIC		98
#define SHUTDOWN_NIC	99
#define STOP_SYSTEM     100
#define GNOM_TX_SEND    101
#define GNOM_MAP_USER_PAGE  102

#define GNOM_REG_MULT_CPU_BUFFERS 110
#define GNOM_UNREG_MULT_CPU_BUFFERS 111

#define TEST_MAP_SINGLE_PAGE 200
#define TEST_UNMAP_SINGLE_PAGE 201
#define TEST_SEND_SINGLE_PACKET 202
#define TEST_CHECK_SEND_COMPLETE 203


/********** Different Configurations to test GNoM Performance and Functionality ***********/
//#define DO_GNOM_TX    // Enable if GNoM handles TX. Disable if PF_RING handles TX
#define SINGLE_GPU_BUFER
#define SINGLE_GPU_PKT_PTR
/******************************************************************************************/


/*********** RX ***********/
// NIC page, buffer, ring defines
#define NUM_GPU_RINGS			1

// 112640 = 220MB of pinned GPU memory. This is the maximum amount on Tesla K20c driver 340.65
#define NUM_GPU_BUF_PER_RING    112640
#define RING_BUF_MULTIPLIER		1
#define RX_PAGE_SZ				1024*64

// NOTE:   GPUDirect requires RX_BUFFER_SZ==2048.
//          NGD on Maxwell requires 1024
#define RX_BUFFER_SZ          2048
//#define RX_BUFFER_SZ          1024

#define NUM_GPU_BUFFERS			NUM_GPU_RINGS*NUM_GPU_BUF_PER_RING*RING_BUF_MULTIPLIER
#define NUM_GPU_PAGES			NUM_GPU_BUFFERS / (RX_PAGE_SZ/RX_BUFFER_SZ)

/*********** TX ***********/
// Currently GNoM does not perform TX through GPUDirect. 
// TX is through PF_RING. However, an unoptimized GPUDirect
// TX path is implemented and can be enabled with DO_GNOM_TX 
// TODO: Verify DO_GNOM_TX still works. 
#define NUM_GPU_TX_RINGS        1
#define NUM_GPU_TX_BUF_PER_RING 8192
#define RING_TX_BUF_MULTIPLIER  32
#define TX_PAGE_SZ              1024*4

#define TX_BUFFER_SZ            1024
#define NUM_GPU_TX_BUFFERS      NUM_GPU_TX_RINGS*NUM_GPU_TX_BUF_PER_RING*RING_TX_BUF_MULTIPLIER
#define NUM_GPU_TX_PAGES        NUM_GPU_TX_BUFFERS / (TX_PAGE_SZ/TX_BUFFER_SZ)

#endif /* GPU_KM_SHARED_H_ */
