##################################################################
# 
# Tayler Hetherington
# 2015
# 	Top-level Makefile for MemcachedGPU, GNoM-KM, and GNoM-ND.
# 	This Makefile checks that the environment is correctly 
# 	configured before building. 
#	
#################################################################

# Static location of GNoM_km set here.
# This can be overridden if needed
GNOM_KM_DIR ?= $(shell pwd)/src/GNoM_km/
export GNOM_KM_DIR

gnom: check_environment
	$(MAKE) -C ./src/GNoM_km/
	$(MAKE) -C ./src/GNoM_nd/src/
	$(MAKE) -C ./src/MemcachedGPU/cuda_kernel_files/
	$(MAKE) -C ./src/MemcachedGPU/

check_environment:
	@if [ ! -n "$(NVIDIA_KERNEL_DIR)" -o ! -n "$(CUDA_INSTALL_DIR)" -o ! -n "$(CUDA_COMMON_DIR)" -o ! -n "$(PF_RING_DIR)" ]; then \
		echo "**GNoM Build Error**: NVIDIA_KERNEL_DIR, CUDA_INSTALL_DIR, CUDA_COMMON_DIR, or PF_RING_DIR environment variables not set"; \
		exit 101; \
	fi

all: 
	$(MAKE) gnom

gnom_km: check_environment
	$(MAKE) -C ./src/GNoM_km/

gnom_nd: check_environment
	$(MAKE) -C ./src/GNoM_nd/src/

memcachedGPU: check_environment
	$(MAKE) -C ./src/MemcachedGPU/cuda_kernel_files/
	$(MAKE) -C ./src/MemcachedGPU/

clean:
	$(MAKE) -C ./src/GNoM_km/ clean
	$(MAKE) -C ./src/GNoM_nd/src/ clean
	$(MAKE) -C ./src/MemcachedGPU/cuda_kernel_files/ clean
	$(MAKE) -C ./src/MemcachedGPU/ clean

