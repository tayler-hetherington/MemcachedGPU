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
 * cuda_context_manager.cpp
 */

extern "C" {
    #include "cuda_context_manager.h"
}
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstring>

#include "gpu_common.h"

/**********************************************/
// Code in this file was borrowed and modified
// from an NVIDIA CUDA sample.
/**********************************************/


//////// Forward Declararions
#define CLEANUP_ON_ERROR(hcuModule, hcuContext, status) \
    if ( hcuModule ) cuModuleUnload( hcuModule ); \
    if ( hcuContext ) cuCtxDetach( hcuContext ); \
    return status;

#define THREAD_QUIT \
    printf("Error\n"); \
    return 0;


// Static paths to CUDA PTX files
#define PTX_FILE_1 "./cuda_kernel_files/gnom_set_assoc_memc.ptx"
#define CUBIN_FILE_1 "./cuda_kernel_files/gnom_set_assoc_memc.cubin"

#define PTX_FILE_2 "./cuda_kernel_files/ngd_gnom_set_assoc_memc.ptx"
#define CUBIN_FILE_2 "./cuda_kernel_files/ngd_gnom_set_assoc_memc.cubin"

#define PTX_FILE_3 "./cuda_kernel_files/gpu_network.ptx"
#define CUBIN_FILE_3 "./cuda_kernel_files/gpu_network.cubin"

#define PTX_FILE_4 "./cuda_kernel_files/background_kernel.ptx"
#define CUBIN_FILE_4 "./cuda_kernel_files/background_kernel.cubin"


bool inline findModulePath(const char *module_file, std::string &module_path, std::string &ptx_source, char *actual_path);
static CUresult InitCUDAContext(CUDAContext *pContext, CUdevice hcuDevice, int deviceID, int n_streams, int gnom_config, int device_num);



extern "C" bool init_GPU(CUDAContext **context, int n_streams, int gnom_config, int device_num){
    CUresult status;
    int device_count = 0;

    CUdevice hcuDevice = device_num;
    int deviceID=device_num;

    status = cuInit(0);
    if(status != CUDA_SUCCESS)
        return false;

    status = cuDeviceGetCount(&device_count);
    printf("%d CUDA device(s) fou"
            "nd\n\n", device_count);

    if (device_count == 0){
        return false;
    }

    int ihThread = 0;
    int ThreadIndex = 0;


    // Print out the available CUDA devices on this system
    for (int iDevice = 0; iDevice < device_count; iDevice++) {
        char szName[256];
        status = cuDeviceGet(&hcuDevice, iDevice);

        if (CUDA_SUCCESS != status)
            return false;

        status = cuDeviceGetName(szName, 256, hcuDevice);

        if (CUDA_SUCCESS != status)
            return false;

        CUdevprop devProps;

        if (CUDA_SUCCESS == cuDeviceGetProperties(&devProps, hcuDevice)){


        	int major = 0, minor = 0;
            checkCudaErrors(cuDeviceComputeCapability(&major, &minor, hcuDevice));
            printf("Device %d: \"%s\" (Compute %d.%d)\n", iDevice, szName, major, minor);
            printf("\tsharedMemPerBlock: %d\n", devProps.sharedMemPerBlock);
            printf("\tconstantMemory   : %d\n", devProps.totalConstantMemory);
            printf("\tregsPerBlock     : %d\n", devProps.regsPerBlock);
            printf("\tclockRate        : %d\n", devProps.clockRate);
            
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, hcuDevice);
            if (devProp.major < 3 || (devProp.major == 3 && devProp.minor < 5)){
                if (devProp.concurrentKernels == 0) {
                    printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
                    printf("  CUDA kernel runs will be serialized\n");
                }else{
                    printf("> GPU does not support HyperQ\n");
                    printf("  CUDA kernel runs will have limited concurrency\n");
                }
               
            }
            printf("> Detected Compute SM %d.%d hardware with %d multi-processors, concurrent kernel supported=%d\n",
                          devProp.major, devProp.minor, devProp.multiProcessorCount, devProp.concurrentKernels); 
            printf("\n");

        }

        int x=0;
        status = cuDeviceGetAttribute(&x, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, hcuDevice);
        printf("Host memory mapping:    %d\n", x);
    }

    if(*context == NULL){ // Context not yet allocated
        *context = new CUDAContext;
    }

    hcuDevice = device_num;
    deviceID = device_num;
    // Initialize "context"
    if(InitCUDAContext(*context, hcuDevice, deviceID, n_streams, gnom_config, device_num) == CUDA_SUCCESS){
        printf("CUDA context initialized successfully...\n");
    }


}

bool inline findModulePath(const char *module_file, std::string &module_path, std::string &ptx_source, char *actual_path){
    if (actual_path){
        module_path = actual_path;
    }
    else{
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty()){
        printf("> findModulePath could not find file: <%s> \n", module_file);
        return false;
    }
    else{
        printf("> findModulePath found file at <%s>\n", module_path.c_str());

        if (module_path.rfind(".ptx") != std::string::npos){
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }
        return true;
    }
}


static CUresult buildCUDAModule( CUmodule *module, CUfunction *function, CUcontext *context,
                                    char *ptx_name, char *cubin_name, 
                                    char *path, char *kernel_function_name){

    std::string module_path, ptx_source;
    CUresult status = CUDA_SUCCESS;

    if (!findModulePath(ptx_name, module_path, ptx_source, path)) {
        if (!findModulePath(cubin_name, module_path, ptx_source, path)){
            fprintf(stderr, "> findModulePath could not find <threadMigration> ptx or cubin\n");
            CLEANUP_ON_ERROR(*module, *context, status);
        }
    }

    printf("Compling kernel at path: %s\n", path);

    if (module_path.rfind(".ptx") != std::string::npos){
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 128;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        status = cuModuleLoadDataEx(module, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
        printf("> PTX JIT log:\n%s\n", jitLogBuffer);

        if(jitOptions)
            delete jitOptions;
        if(jitOptVals)
            delete jitOptVals;
        if(jitLogBuffer)
            delete jitLogBuffer;

    }
    else{
        status = cuModuleLoad(module, module_path.c_str());

        if (CUDA_SUCCESS != status) {
            fprintf(stderr, "cuModuleLoad failed %d\n", status);
            CLEANUP_ON_ERROR(*module, *context, status);
        }
    }

    status = cuModuleGetFunction(function, *module, kernel_function_name);
    if (CUDA_SUCCESS != status){
        fprintf(stderr, "cuModuleGetFunction (%s) failed %d\n", kernel_function_name, status);
        CLEANUP_ON_ERROR(*module, *context, status);
    }

    module_path.clear();
    ptx_source.clear();

    return CUDA_SUCCESS;
}


static CUresult InitCUDAContext(CUDAContext *pContext, CUdevice hcuDevice, int deviceID, int n_streams, int gnom_config, int device_num){
    CUcontext  hcuContext  = 0;
    CUmodule   hcuModule   = 0;
    CUfunction hcuFunction = 0;
    CUfunction set_function = 0;
    CUdevprop devProps;

    CUmodule   network_module   = 0;
    CUfunction network_function = 0;

    CUmodule    background_module = 0;
    CUfunction  background_function = 0;


    CUresult status = cuCtxCreate(&hcuContext, CU_CTX_MAP_HOST, hcuDevice);

    if (CUDA_SUCCESS != status) {
        fprintf(stderr, "cuCtxCreate for <Thread=%d> failed %d\n",
                pContext->threadNum, status);
        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
    }

    status = CUDA_ERROR_INVALID_IMAGE;

    if (CUDA_SUCCESS != cuDeviceGetProperties(&devProps, hcuDevice)) {
        printf("cuDeviceGetProperties FAILED\n");
        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
    }


    printf("Selecting Device # %d\n", hcuDevice);

    char szName[256];
    status = cuDeviceGet(&hcuDevice, device_num);

    status = cuDeviceGetName(szName, 256, hcuDevice);

    if (CUDA_SUCCESS == cuDeviceGetProperties(&devProps, hcuDevice)){
        int major = 0, minor = 0;
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, hcuDevice));
        printf("Device %d: \"%s\" (Compute %d.%d)\n", device_num, szName, major, minor);
        printf("\tsharedMemPerBlock: %d\n", devProps.sharedMemPerBlock);
        printf("\tconstantMemory   : %d\n", devProps.totalConstantMemory);
        printf("\tregsPerBlock     : %d\n", devProps.regsPerBlock);
        printf("\tclockRate        : %d\n", devProps.clockRate);

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, hcuDevice);
        if (devProp.major < 3 || (devProp.major == 3 && devProp.minor < 5)){
            if (devProp.concurrentKernels == 0) {
                printf("> GPU does not support concurrent kernel execution (SM 3.5 or higher required)\n");
                printf("  CUDA kernel runs will be serialized\n");
            }else{
                printf("> GPU does not support HyperQ\n");
                printf("  CUDA kernel runs will have limited concurrency\n");
            }

        }
        printf("> Detected Compute SM %d.%d hardware with %d multi-processors, concurrent kernel supported=%d\n",
                      devProp.major, devProp.minor, devProp.multiProcessorCount, devProp.concurrentKernels);
        printf("\n");

    }

    char *actual_path = NULL;

    char *gnom_path =  (char *)PTX_FILE_1;
    char *ngd_path = (char *)PTX_FILE_2;
    char *network_only_path = (char *)PTX_FILE_3;
    char *background_path = (char *)PTX_FILE_4;

    switch(gnom_config){
        case 0:
            actual_path = gnom_path;
            break;
        case 1: 
            actual_path = ngd_path;
            break;
        default:
            printf("Error: Unimplemented gnom_config in CUDA Context manager\n");
            abort();
    }


    status = buildCUDAModule( &hcuModule, &hcuFunction, &hcuContext, 
            (char *)PTX_FILE_1, (char *)CUBIN_FILE_1, actual_path, (char *)"memcached_GET_kernel");
    if (CUDA_SUCCESS != status){
        fprintf(stderr, "buildCUDAModule Failed %d\n", status);
        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
    }
    printf("GET kernel complete\n");

    // Second function from main hcuModule
    status = cuModuleGetFunction(&set_function, hcuModule, "memcached_SET_kernel");
    if (CUDA_SUCCESS != status){
        fprintf(stderr, "cuModuleGetFunction (memcached_SET_kernel) failed %d\n", status);
        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
    }
    printf("SET kernel complete\n");


    status = buildCUDAModule( &network_module, &network_function, &hcuContext, 
            (char *)PTX_FILE_3, (char *)CUBIN_FILE_3, network_only_path, (char *)"network_kernel");

    if (CUDA_SUCCESS != status){
        fprintf(stderr, "buildCUDAModule Failed %d\n", status);
        CLEANUP_ON_ERROR(network_module, hcuContext, status);
    }
    printf("Network kernel complete\n");

    status = buildCUDAModule( &background_module, &background_function, &hcuContext, 
            (char *)PTX_FILE_4, (char *)CUBIN_FILE_4, background_path, (char *)"background_kernel");

    if (CUDA_SUCCESS != status){
        fprintf(stderr, "buildCUDAModule Failed %d\n", status);
        CLEANUP_ON_ERROR(background_module, hcuContext, status);
    }
    printf("Background kernel complete\n");


    // Create streams
    CUstream *streams = (CUstream *)malloc((n_streams + 1) * sizeof(CUstream));
    for(unsigned i=0; i<(n_streams+1); ++i){
        status = cuStreamCreate(&streams[i], 0);
        if(CUDA_SUCCESS != status){
            fprintf(stderr, "cuStreamCreate failed : %d", status);
            CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
        }
    }

    // Set cache config to favour shared memory
//    status = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED);
//    if (CUDA_SUCCESS != status) {
//        fprintf(stderr, "cuCtxSetCacheConfig failed %d\n", status);
//        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
//    }

    // Here we must release the CUDA context from the thread context
    status = cuCtxPopCurrent(NULL);

    if (CUDA_SUCCESS != status){
        fprintf(stderr, "cuCtxPopCurrent failed %d\n", status);
        CLEANUP_ON_ERROR(hcuModule, hcuContext, status);
    }

    pContext->hcuContext  = hcuContext;
    pContext->hcuModule   = hcuModule;
    pContext->hcuFunction = hcuFunction;
    pContext->set_function = set_function;
    pContext->deviceID    = deviceID;
    pContext->streams     = streams;
    pContext->n_streams   = n_streams+1; // +1 for separate SET stream

    pContext->network_module = network_module;
    pContext->network_function = network_function;
    pContext->background_module = background_module;
    pContext->background_function = background_function;

    return CUDA_SUCCESS;
}



bool delete_GPU(CUDAContext *context){
    
    // Delete Streams
    for(unsigned i=0; i<context->n_streams+1; ++i){
        cuStreamDestroy(context->streams[i]);
    }
    free(context->streams);

    // Delete Context
    cuCtxDestroy(context->hcuContext);

    //Delete context object wrapper
    delete context;
}
