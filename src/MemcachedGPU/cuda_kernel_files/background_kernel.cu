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
 * background_kernel.cu
 */


// CUDA utilities and system includes
#ifndef __CUDA_VERSION__
//#define __
#endif

#include <cuda_runtime.h>
#include <host_defines.h>
#include <device_launch_parameters.h>

#include <stdio.h>

// 16 Blocks
// 1024 threads / block
// Array size = 16K min
// Array size * 4 = 64K

#define ARRAY_MULTIPLIER 4 

// Simple vector multiplication kernel. Performs loads, stores, and computations on a large amount of data. 
extern "C" __global__ void background_kernel( int *in_A, int *in_B, int *out_C, int array_length, int num_iterations, int dummy0){       
    int i=0;
    int j=0;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int total_num_threads = gridDim.x*blockDim.x;
    
    int index = 0;
    for(i=0; i<num_iterations; ++i){
        for(j=0; j<ARRAY_MULTIPLIER; ++j){
            index = (tid) + (j*total_num_threads) + (i*dummy0);
            out_C[index] = in_A[index] * in_B[index];
        }
    }
}












































