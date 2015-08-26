#!/bin/bash

FAMILY=ixgbe

echo "Loading GNoM modules and running GNoM + MemcachedGPU..." 

# Remove old modules (if loaded)
# Remove GNoM-ND
rmmod ixgbe
# Remove PF_RING
rmmod pf_ring
# Remove GNoM-KM
rmmod gpu_km

# Hugepages for PF_RING
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

mkdir /mnt/huge
mount -t hugetlbfs nodev /mnt/huge

# GNoM-KM dir
GNOM_KM_DIR=../src/GNoM_km/

# GNoM-ND dir
GNOM_ND_DIR=../src/GNoM_nd/src/


#######################
##### Load Modules ####
#######################

# Load GNoM-KM
echo "Loading GNoM-KM..."
insmod $GNOM_KM_DIR/gpu_km.ko


# Load PF_RING module (requires PF_RING compiled and location set with PF_RING_DIR env_var)
echo "Loading PF_RING..."
insmod $PF_RING_DIR/kernel/pf_ring.ko


# Load GNoM-ND
echo "Loading GNoM-ND..."

# 2 GNoM-post threads
#insmod ./ixgbe.ko MQ=0,1 RSS=1,2 FdirMode=0,0 FdirPballoc=1,1

# 4 GNoM-post threads
insmod $GNOM_ND_DIR/ixgbe.ko MQ=0,1 RSS=1,4 FdirMode=0,0 FdirPballoc=1,1

# 8 GNoM-post threads
#insmod ./ixgbe.ko MQ=0,1 RSS=1,8 FdirMode=0,0 FdirPballoc=1,1

# Config when using GNoM for TX
#insmod ./ixgbe.ko MQ=1,1 RSS=2,2 FdirMode=2,0 FdirPballoc=3,1

sleep 1


INTERFACES=$(cat /proc/net/dev|grep ':'|grep -v 'lo'|grep -v 'sit'|awk -F":" '{print $1}'|tr -d ' ')
for IF in $INTERFACES ; do
	TOCONFIG=$(ethtool -i $IF|grep $FAMILY|wc -l)
        if [ "$TOCONFIG" -eq 1 ]; then
		printf "Configuring %s\n" "$IF"
		ifconfig $IF up
		sleep 1
	fi
done

