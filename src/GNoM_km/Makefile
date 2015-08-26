# If KERNELRELEASE is defined, we've been invoked from the
# kernel build system and can use its language.

# Tested with NVIDIA driver 340.65
KBUILD_EXTRA_SYMBOLS := $(NVIDIA_KERNEL_DIR)/Module.symvers


ifndef NVIDIA_KERNEL_DIR
$(error NVIDIA_KERNEL_DIR environment variable is not set. This should be set to the NVIDIA driver kernel directory (e.g., <...>/NVIDIA-Linux-x86_64-340.65/kernel/))
endif

# Adding the NVIDIA kernel directory to the include list for nv-p2p.h
EXTRA_CFLAGS += -I$(NVIDIA_KERNEL_DIR)

ifneq ($(KERNELRELEASE),)
    obj-m := gpu_km.o
# Otherwise we were called directly from the command
# line; invoke the kernel build system.
else
KERNELDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)


default:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
endif

