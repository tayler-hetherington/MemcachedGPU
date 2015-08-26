echo "Killing any memcached instances..."
pkill memcached

echo "Removing GNoM-ND..."
rmmod ixgbe
echo "Removing PF_RING..."
rmmod pf_ring
echo "Removing GNoM_KM..."
rmmod gpu_km

echo "Unmounting Hugetlbfs..."
umount -v -f /mnt/huge 
echo "Complete!"
