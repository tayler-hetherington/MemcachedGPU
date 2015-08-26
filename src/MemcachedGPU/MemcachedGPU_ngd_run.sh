# Configures 
#   Setup 1024 MB of Memcached storage (-m 1024)
#   TCP port for SETs to 9999 (-p 9999)
#   GPU to device 0 (-g 0)
#   GNoM configuration to use Non-GPUDirect (NGD) (-e 1)


CUR_DIR=$(dirname $0)
$CUR_DIR/memcached -m 1024 -p 9999 -g 0 -e 1 &
