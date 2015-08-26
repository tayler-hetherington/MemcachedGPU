NVCC := nvcc
PTX_FILE_1 := gnom_set_assoc_memc.ptx
PTX_FILE_2 := ngd_gnom_set_assoc_memc.ptx
PTX_FILE_3 := gpu_network.ptx
PTX_FILE_4 := background_kernel.ptx

GENCODE_SM30    := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM30)
NVCCFLAGS := -m64

all: $(PTX_FILE_1) $(PTX_FILE_2) $(PTX_FILE_3) $(PTX_FILE_4)

$(PTX_FILE_1): gnom_set_assoc_memc.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc -o $@ -ptx $<

$(PTX_FILE_2): ngd_gnom_set_assoc_memc.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc -o $@ -ptx $<

$(PTX_FILE_3): cuda_network.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc -o $@ -ptx $<

$(PTX_FILE_4): background_kernel.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc -o $@ -ptx $<

clean:
	rm *.ptx
