# MemcachedGPU

This README provides information on how to configure, build, and run GNoM and MemcachedGPU.

# Publications

MemcachedGPU: Scaling-up Scale-out Key-value Stores - SoCC'15 (http://ece.ubc.ca/~taylerh/doc/MemcachedGPU_SoCC15.pdf)

# Building GNoM and MemcachedGPU

Prerequisites:

    - CUDA 5.5 (Note: The GNoM and MemcachedGPU framework has only been tested with CUDA 5.5). 

    - CUDA SDK Samples (e.g., NVIDIA_CUDA-5.5_Samples/)

    - PF_RING version 6.0.0 (This was the version used)

    - Libevent.

    - GPUDirect setup: NVIDIA driver kernel (Follow the information here to setup GPUDirect http://docs.nvidia.com/cuda/gpudirect-rdma/index.html). 

**NOTE**: The Ethernet device names are currently hardcoded in this version of GNoM and MemcachedGPU. You will need to change them to match your system before building or running GNoM and MemcachedGPU. I will be fixing this in future versions. 

    - In my configuration, I used port 0 (eth2) for receiving, and port 1 (eth3) for sending. 

    - You will need to change the following locations

        - RX

            - <root_dir>/src/MemcachedGPU/cuda_stream_manager.cpp: lines 2417->2419 (Change all eth2@# to match the Ethernet port you wish to receive at). 

            - <root_dir>/src/GNoM_nd/src/gnom_ixgbe.h: lines 50 and 53 (Change the "GNOM_RX_DEVICE_PCI_NAME" to match your port device name and the GNOM_RX_NAME, e.g. "0000:04:00.0" and "eth2")          

        - TX

            - <root_dir>/src/MemcachedGPU/cuda_stream_manager.cpp: lines 344->345 (Change all eth3@# to match the Ethernet port you wish to send from). 

            - <root_dir>/src/GNoM_nd/src/gnom_ixgbe.h: lines 51 and 54 (Change the "GNOM_TX_DEVICE_PCI_NAME" to match your port device name and the GNOM_TX_NAME, e.g. "0000:04:00.1" and "eth3")



Before building GNoM and MemcachedGPU, the following environment variables must be set:

    - CUDA_INSTALL_DIR: This must be set to the CUDA installation directory (e.g., /home/<user>/cuda5.5/cuda/) 

    - CUDA_COMMON_DIR: This must be set to the CUDA common directory in the SDK samples (e.g., /home/<user>/NVIDIA_CUDA-4.4_Samples/common/)

    - PF_RING_DIR: This must be set to the toplevel PF_RING installation directory (e.g., /home/<user>/PF_RING/).

    - (Optional) GNOM_KM_DIR: This is set automatically by the Makefiles, however, if you are moving directories or 
      files from their default location, you should set this environment variable to point to the GNoM_km/ directory.   

The Makefile will fail if any of the environment variables have not been correctly set. There is an example setup environment file in MemcachedGPU/scripts/example_setup_env.sh. Modify this file to match your directories and then run "source example_setup_env.sh"

First you need to configure MemcachedGPU

    - Change to the MemcachedGPU directory (e.g., <root_dir>/MemcachedGPU/src/MemcachedGPU/).

    - Run "automake"

    - Run "./configure --with-libevent=<path to libevent install directory>"

This will configure MemcachedGPU on your system. 

Then, simply run 'make' in the top-level directory (same as this README) to build. 

    - This will build GNoM-KM, GNoM-ND, and GNoM-User (GNoM-Host and GNoM-Device). 

# Running MemcachedGPU

The information below describes how to load the GNoM-KM, GNoM-ND, and PF_RING modules, and run GNoM/MemcachedGPU.
I will continue to make this process more automated. GNoM requires privileged permissions and will need to be run with "sudo".

    - Change to the scripts directory (e.g., <root_dir>/MemcachedGPU/scripts)

    - Make sure you've already modified and run the "example_setup_env.sh" script to configure your environment and have already compiled everything.

    - Load the modules with the following command

        - "sudo -E ./gnom_load_and_run.sh"

    - Then change directories to the MemcachedGPU code (e.g., cd <root_dir>/src/MemcachedGPU/)

        - Look at the "MemcachedGPU_run.sh" and "MemcachedGPU_ngd_run.sh" scripts. These scripts run MemcachedGPU with either the GPUDirect or Non-GPUDirect frameworks. These scripts describe the parameters used to select a GPU device and the configuration.

        - Run the following command to run MemcachedGPU with GPUDirect

            - "sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./MemcachedGPU_run.sh"

    - MemcachedGPU should be up and running and waiting for SET requests over TCP port 9999, and GET requests over UDP port 9960. 

    - To kill MemcachedGPU, run the following command

        - "sudo pkill MemcachedGPU".

        - NOTE: The GNoM-pre thread may be stuck in GNoM-km waiting for an incomming RX packet batch. You can either send UDP GET packets to GNoM until a batch is received and the thread exits, or you can wait for the timeout (~20 seconds). 

    - Unload the modules with the script "unload_modules.sh" in <root_dir>/scripts/. 

