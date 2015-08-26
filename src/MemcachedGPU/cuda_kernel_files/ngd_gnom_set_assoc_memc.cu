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
 * cuda_gpu_nom_memcached.cu
 */

// Set Associative version of the hash table

// CUDA utilities and system includes
#ifndef __CUDA_VERSION__
//#define __
#endif

#include <cuda_runtime.h>
#include <host_defines.h>
#include <device_launch_parameters.h>

#include <stdio.h>

//#define DEBUG // Uncomment to enable debugging

// If this is set, MemcachedGPU sends back the 8Byte Memcached header with the response
//#define LATENCY_MEASURE

// If this is set, the response packet is a constant size (RESPONSE_SIZE) independent from the Memcached packet
// The packet header/checksum can be computed earlier in parallel with the Memcached lookup.
//#define CONSTANT_RESPONSE_SIZE
#define RESPONSE_SIZE 72 //80  // 72 for peak throughput, 80 for latency test


#define USE_KEY_HASH
#define KEY_HASH_MASK   0x0000000FF
#define SET_ASSOC_SIZE      16

#define RESPONSE_HDR_STRIDE 256

#define NETWORK_PKT_SIZE    42

/*************************************/
#define REQUEST_GROUP_SIZE		128 // Don't change // Number of requests per group (subset of batch)
#define MAX_THREADS_PER_BLOCK	256 // Number of threads per request group
#define NUM_REQUESTS_PER_GROUP  256 // Do not change
/*************************************/


// This should be changed to match the number of requests per batch in GNoM_KM and GNoM_User
// (Should match NUM_REQUESTS_PER_BATCH in GNoM_km/gpu_km_shared.h)
#define NUM_REQUESTS_PER_BATCH  512 //256 

#define NUM_THREADS_PER_GROUP   NUM_REQUESTS_PER_GROUP*2 // NUM_REQUESTS_PER_BATCH*2
#define NUM_GROUPS NUM_REQUESTS_PER_BATCH / NUM_REQUESTS_PER_GROUP
/*************************************/

#define SINGLE_GPU_PKT_PTR

// TODO: The current Non-GPUDirect implementation requires all requests to be the same 
//       size. It batches requests at constant offsets. There should be a header in the
//       buffer before the first patcket speciying the packet offsets. 

#define RX_BUFFER_SZ 72     // 16 Byte key
//#define RX_BUFFER_SZ 88   // 32 Byte key
//#define RX_BUFFER_SZ 120  // 64 Byte key
//#define RX_BUFFER_SZ 184  // 128 Byte key



#define UDP_PORT		9960
#define ETH_ALEN		6
#define IPPROTO_UDP		17

// Smaller max key size for testing.
#define MAX_KEY_SIZE	140 //250


#define UNLOCKED			0	// No lock set
#define SHARED_LOCK			1	// GET request(s) have the item locked
#define PRIVATE_LOCK		2	// SET request has the item locked. Only a single PRIVATE_LOCK can be obtained at a time.


#define G_HTONS(val) (u_int16_t) ((((u_int16_t)val >> 8) & 0x00FF ) | (((u_int16_t)val << 8) & 0xFF00) )
#define G_NTOHS(val) (G_HTONS(val))

#define hashsize(n) ((unsigned int)1<<(n))
#define hashmask(n) (hashsize(n)-1)

typedef unsigned int rel_time_t;

// Placeholder for Memcached item pointers
typedef void item;

typedef struct _ether_header{
  u_int8_t  ether_dhost[ETH_ALEN];	/* destination eth addr	*/
  u_int8_t  ether_shost[ETH_ALEN];	/* source ether addr	*/
  u_int16_t ether_type;		        /* packet type ID field	*/
}ether_header;


typedef struct _ip_header {
  u_int8_t	version;			/* version */			// Version+ihl = 8 bits, so replace ihl with 8bit version
  //u_int32_t ihl:4;			/* header length */

  u_int8_t	tos;			    /* type of service */
  u_int16_t	tot_len;			/* total length */
  u_int16_t	id;			        /* identification */
  u_int16_t	frag_off;			/* fragment offset field */
  u_int8_t	ttl;			    /* time to live */
  u_int8_t	protocol;			/* protocol */
  u_int16_t	check;			    /* checksum */

  u_int16_t saddr1;             /* source and dest address */
  u_int16_t saddr2;
  u_int16_t daddr1;
  u_int16_t daddr2;

}ip_header;

typedef struct _udp_header {
  u_int16_t	source;		/* source port */
  u_int16_t	dest;		/* destination port */
  u_int16_t	len;		/* udp length */
  u_int16_t	check;		/* udp checksum */
}udp_header;

typedef struct _memc_hdr_{
	u_int8_t hdr[14]; // Only 8 Bytes, but padding an extra 4 bytes for memcpy purposes
}memc_hdr;

typedef struct _pkt_memc_hdr_{
	ether_header eh;
	ip_header iph;
	udp_header udp;
	memc_hdr mch;
}pkt_memc_hdr;

typedef struct _pkt_res_memc_hdr_{
    ether_header eh;
    ip_header iph;
    udp_header udp;
    char valstr_key[RESPONSE_HDR_STRIDE - NETWORK_PKT_SIZE];
}pkt_res_memc_hdr;


typedef struct _mod_pkt_info_{
	item *it; 	            // CPU VA pointer to found item
    unsigned pkt_length;    // Total length of response packet => Packet UDP header + "VALUE " + key + suffix + data (with "\r\n")
	int hv;		// Hash value
	int is_get_req;
	pkt_memc_hdr nmch; // Packet header + memc 8 Byte header
}mod_pkt_info;

typedef unsigned char uint8_t;

typedef struct _key_ {
	unsigned key_len;
	char key[MAX_KEY_SIZE];
} _key_;


// Forward declarations
__device__ int d_memcmp(const void *key1, const void *key2, int num){
	const unsigned *p1 = (const unsigned* )key1;
	const unsigned *p2 = (const unsigned* )key2;

	int main_loop = num / sizeof(int);
	int extra_loop = num % sizeof(int);

	for(unsigned i=0; i<main_loop; i++){
		unsigned diff = *(p1 + i) - *(p2 + i);
		if( diff != 0){
            return 0;
        }
	}

    const char * p12 = ( const char * )key1;
    const char * p22 = (const char*)key2;

    for(unsigned i=main_loop*sizeof(int); i<extra_loop+main_loop*sizeof(int); i++){
	    unsigned char diff = *( p12 + i ) - *( p22 + i );
	    if( diff != 0){
            return 0;
        }
	}

	return 1;
}

// NOTE: This requires key lengths to be in increments 4 bytes
__device__ int fast_memcmp(const void *key1, const void *key2, int num){

    const unsigned *p1 = (const unsigned* )key1;
    const unsigned *p2 = (const unsigned* )key2;

    int main_loop = num / sizeof(int);

    for(unsigned i=0; i<main_loop; i++){
        if(*(p1+i) != *(p2+i)){
            return 0;
        }
    }

    return 1;
}

// Compare char by char
__device__ int slow_memcmp(const char *key1, const char *key2, int num){
    unsigned i=0;
    int flag = 1;
    for(i=0; i<num; i++){
        if(key1[i] != key2[i]){
            flag = 0;
            break;
        }
    }
    return flag;
}

/***********************************************/
/***********************************************/
// Bob Jenkin's hash from baseline Memcached
/***********************************************/
/***********************************************/
#define rot(x,k) (((x)<<(k)) ^ ((x)>>(32-(k))))

#define memcached_mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
   c ^= b; c -= rot(b,14); \
   a ^= c; a -= rot(c,11); \
   b ^= a; b -= rot(a,25); \
   c ^= b; c -= rot(b,16); \
   a ^= c; a -= rot(c,4);  \
   b ^= a; b -= rot(a,14); \
   c ^= b; c -= rot(b,24); \
}

__device__ unsigned int hash(	char const * key,       /* the key to hash */
					size_t length,    /* length of the key */
					const unsigned int initval   /* initval */){

  unsigned int a,b,c;                                          /* internal state */
  union { const char *ptr; size_t i; } u;     /* needed for Mac Powerbook G4 */

  /* Set up the internal state */
  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  u.ptr = key;
  if (((u.i & 0x3) == 0)) {
	   unsigned int const * k = ( unsigned int const *)key;

    /*------ all but last block: aligned reads and affect 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      b += k[1];
      c += k[2];
      memcached_mix(a,b,c);
      length -= 12;
      k += 3;
    }

    switch(length)
    {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;  /* zero length strings require no mixing */
    }
 } else if (((u.i & 0x1) == 0)) {
	  unsigned short const * k = (unsigned short const *)key;                           /* read 16-bit chunks */
	  unsigned char const * k8;

   /*--------------- all but last block: aligned reads and different mixing */
    while (length > 12)
    {
      a += k[0] + (((unsigned int)k[1])<<16);
      b += k[2] + (((unsigned int)k[3])<<16);
      c += k[4] + (((unsigned int)k[5])<<16);
      memcached_mix(a,b,c);
      length -= 12;
      k += 6;
    }

    /*----------------------------- handle the last (probably partial) block */
    k8 = (  unsigned char const *)k;
    switch(length)
    {
    case 12: c+=k[4]+(((unsigned int)k[5])<<16);
             b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 11: c+=((unsigned int)k8[10])<<16;     /* @fallthrough */
    /* no break */
    case 10: c+=k[4];                       /* @fallthrough@ */
             b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 9 : c+=k8[8];                      /* @fallthrough */
    case 8 : b+=k[2]+(((unsigned int)k[3])<<16);
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 7 : b+=((unsigned int)k8[6])<<16;      /* @fallthrough */
    case 6 : b+=k[2];
             a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 5 : b+=k8[4];                      /* @fallthrough */
    case 4 : a+=k[0]+(((unsigned int)k[1])<<16);
             break;
    case 3 : a+=((unsigned int)k8[2])<<16;      /* @fallthrough */
    case 2 : a+=k[0];
             break;
    case 1 : a+=k8[0];
             break;
    case 0 : return c;  /* zero length strings require no mixing */
    }

  } else {                        /* need to read the key one byte at a time */
	   unsigned char const * k = ( unsigned char const *)key;

    /*--------------- all but the last block: affect some 32 bits of (a,b,c) */
    while (length > 12)
    {
      a += k[0];
      a += ((unsigned int)k[1])<<8;
      a += ((unsigned int)k[2])<<16;
      a += ((unsigned int)k[3])<<24;
      b += k[4];
      b += ((unsigned int)k[5])<<8;
      b += ((unsigned int)k[6])<<16;
      b += ((unsigned int)k[7])<<24;
      c += k[8];
      c += ((unsigned int)k[9])<<8;
      c += ((unsigned int)k[10])<<16;
      c += ((unsigned int)k[11])<<24;
      memcached_mix(a,b,c);
      length -= 12;
      k += 12;
    }

    /*-------------------------------- last block: affect all 32 bits of (c) */
    switch(length)                   /* all the case statements fall through */
    {
    case 12: c+=((unsigned int)k[11])<<24;
    case 11: c+=((unsigned int)k[10])<<16;
    case 10: c+=((unsigned int)k[9])<<8;
    case 9 : c+=k[8];
    case 8 : b+=((unsigned int)k[7])<<24;
    case 7 : b+=((unsigned int)k[6])<<16;
    case 6 : b+=((unsigned int)k[5])<<8;
    case 5 : b+=k[4];
    case 4 : a+=((unsigned int)k[3])<<24;
    case 3 : a+=((unsigned int)k[2])<<16;
    case 2 : a+=((unsigned int)k[1])<<8;
    case 1 : a+=k[0];
             break;
    case 0 : return c;  /* zero length strings require no mixing */
    }
  }

  final(a,b,c);
  return c;             /* zero length strings require no mixing */
}
/***********************************************/
/***********************************************/

// This checksum skips the ip_header length field, but adds up everything else.
// Later we can add in the length. Used to overlap independent computation to
// reduce processing latency
__device__ int partial_cksum(unsigned char *buf, unsigned nbytes, int sum) {
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    if(i != 2){ // Bytes 2&3 are the IP header length field, skip it
        sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + i)));
        /* Add carry. */
        if(sum > 0xFFFF)
            sum -= 0xFFFF;
    }
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}


// Only add up the ip header length once we know the response packet size
__device__ int cksum_hdr_len_only(unsigned char *buf, int sum){

    sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + 2)));
    if(sum > 0xFFFF)
        sum -= 0xFFFF;

    return sum;
}

// Full checksum
/*
 * Checksum routine for Internet Protocol family headers (C Version)
 *
 * Borrowed from DHCPd
 */
__device__ int in_cksum(unsigned char *buf, unsigned nbytes, int sum) {
  uint i;

  /* Checksum all the pairs of bytes first... */
  for (i = 0; i < (nbytes & ~1U); i += 2) {
    sum += (u_int16_t) G_NTOHS(*((u_int16_t *)(buf + i)));
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  /* If there's a single byte left over, checksum it, too.   Network
     byte order is big-endian, so the remaining byte is the high byte. */
  if(i < nbytes) {
    sum += buf [i] << 8;
    /* Add carry. */
    if(sum > 0xFFFF)
      sum -= 0xFFFF;
  }

  return sum;
}

/* ******************************************* */

__device__ int wrapsum (u_int32_t sum) {
  sum = ~sum & 0xFFFF;
  return G_NTOHS(sum);
}

/* ******************************************* */

typedef struct _gpu_primary_hashtable_{
	void *item_ptr;
	rel_time_t last_accessed_time;
	unsigned valid;
#ifdef USE_KEY_HASH
    unsigned key_hash;          // 8-bit key hash - using 4 bytes to keep everything aligned
#endif
    unsigned key_length;
    unsigned pkt_length;
	char key[MAX_KEY_SIZE];
}gpu_primary_hashtable;

typedef struct _gpu_set_req_{
	void *item_ptr;
	unsigned init_hv;
	unsigned key_length;
    unsigned pkt_length;
	char key[MAX_KEY_SIZE];
}gpu_set_req;

typedef struct _gpu_set_res_{
	int host_signal;
	int is_evicted;
    int is_last_get;
	unsigned evicted_hv;
    unsigned evicted_lru_timestamp;
	void *evitcted_ptr;
}gpu_set_res;

// Forward declarations
__device__ void mod_parse_pkt(  unsigned long long first_RX_buffer_ptr,
                                int local_tid,
                                int logical_tid,
                                int thread_type,
                                mod_pkt_info *mpi,
                                _key_ *g_key);

__device__ int mod_process_get_request( mod_pkt_info *mpi,
                                        int hashpower,
                                        rel_time_t time,
                                        volatile gpu_primary_hashtable *g_primary_hashtable,
                                        _key_ *g_key,
                                        int *gpu_hash_lock);

__device__ void mod_create_response_header(mod_pkt_info *mpi, int helper_tid);
__device__ void mod_populate_response(size_t *res_mem, mod_pkt_info *mpi, int tid, int helper_tid, int group_id, int *item_is_found, unsigned thread_type, int cta_id, _key_ *g_key);

extern "C" __global__ void memcached_SET_kernel(int *req_mem,
												int *res_mem,
												int hashpower,					// Memcached hashpower
												unsigned int *gpu_hashtable,	// GPU resident Memcached hashtable
												int *gpu_hash_lock,	// GPU resident locks for hashtable
												rel_time_t timestamp){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int ret=0;

    __shared__ unsigned hv;
    __shared__ unsigned set_index;
    __shared__ unsigned set_hv_index;
    __shared__ unsigned insert_hv_index;  
    __shared__ unsigned key_hash_t;

    __shared__ unsigned evict_lru_timestamp;

    volatile gpu_primary_hashtable *g_primary_hashtable = (volatile gpu_primary_hashtable *)gpu_hashtable;

	gpu_set_req *m_gph = (gpu_set_req *)req_mem;
    gpu_set_res *m_gsr = (gpu_set_res *)res_mem;

    volatile gpu_primary_hashtable *temp_gph;
    volatile gpu_primary_hashtable *gph;

	int oldest_item_hv 		= -1;
	size_t oldest_item_time = 0xFFFFFFFFFFFFFFFF;
	int free_found 		    = 0;

	int is_locked = 0;
	int old_lock_val = 0;

	unsigned num_sets = hashsize(hashpower) / SET_ASSOC_SIZE;


#ifdef DEBUG
	unsigned hv = 0;
	if(tid==0){
	    hv = hash(m_gph->key, m_gph->key_length, 0);
	    if(hv != m_gph->init_hv){
	        printf("HASH VALUES NOT EQUAL!!\n");
	    }
	}
#endif

    m_gsr->is_evicted = 0; // Set initial to SET eviction. May end up finding a free entry (0) or being a SET hit (2)

	// Set Assoc Hashing - Search for a free spot within the set from init_hv
	if(tid == 0){
		hv = m_gph->init_hv;                        // Grab the hash value from the CPU calculation
		set_index = hv % num_sets;                  // Calculate the set index for this hash value
		set_hv_index = set_index*SET_ASSOC_SIZE;    // Move to the correct location in the hash table for this set

        key_hash_t =  hv & KEY_HASH_MASK; // Calcualte the hash mask

		// Lock the current set
		while(!is_locked){
		    old_lock_val = atomicCAS(&gpu_hash_lock[set_index], 0, -1);
		    if(old_lock_val == UNLOCKED){
		        is_locked = 1;
		    }
		}

		for(unsigned i=0; i<SET_ASSOC_SIZE; ++i){
		    temp_gph = (volatile gpu_primary_hashtable *)&g_primary_hashtable[set_hv_index + i]; // Index into the hashtable at this set

		    if(temp_gph->valid > 0){ // This hash location is already occupied, check the next location

                // First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
                if(temp_gph->key_hash == key_hash_t){
                    // If key hash matches, check complete key
                    ret = fast_memcmp((const void *)m_gph->key, (const void *)temp_gph->key, m_gph->key_length);
 
                    if(ret == 1){
                        // If matches, select this entry to overwrite. Set matching key-value pair to evict.
                        // This is required to ensure correct ordering on the CPU post processing

                        // Treat this the same as an LRU evict
                        oldest_item_time = temp_gph->last_accessed_time;
                        oldest_item_hv = (set_hv_index+i);
                        free_found = 0;
                        m_gsr->is_evicted = 2; // Set to SET hit
                        break;
                    }
                }

                // If no hit, update LRU status for this set                
                if((temp_gph->last_accessed_time < oldest_item_time) || (oldest_item_hv == -1)){
                    oldest_item_time = temp_gph->last_accessed_time;
                    oldest_item_hv = (set_hv_index+i);
                }
            }else{
                // No need to search the whole set if an invalid entry is found
                free_found = 1;
                insert_hv_index = (set_hv_index + i);
                break;
            }

		}

		if(!free_found){
		    // Didn't find any free spots... Need to evict an item with the oldest timestamp within the set
		    insert_hv_index = oldest_item_hv;
            evict_lru_timestamp = oldest_item_time;
            if(m_gsr->is_evicted == 0){
                m_gsr->is_evicted = 1;
            }
		}

	}
    __syncthreads();
    __threadfence();

    gph = (volatile gpu_primary_hashtable *)&g_primary_hashtable[insert_hv_index]; // Index into the hashtable

    unsigned int *temp_key_src = (unsigned int *)m_gph->key;
    unsigned int *temp_key_dst = (unsigned int *)gph->key;

    // Block memory copy with all threads in the warp (max key size of 128 with this code)
    if(tid < 32){
        temp_key_dst[tid] = temp_key_src[tid]; // Copy the key over (Maybe overwriting previous key)
    }

    __syncthreads();
    __threadfence();

	if(tid == 0){
		if(!free_found){
			m_gsr->evicted_hv = oldest_item_hv;
			m_gsr->evitcted_ptr = gph->item_ptr;
            m_gsr->evicted_lru_timestamp = evict_lru_timestamp;
		}

		// Set 8-bit key hash
		gph->key_hash = hv & KEY_HASH_MASK;

		gph->item_ptr = m_gph->item_ptr;

        gph->key_length = m_gph->key_length;
        gph->pkt_length = m_gph->pkt_length;

        // Record whether the last access was a SET or GET request 
        if(gph->valid == 1){
            m_gsr->is_last_get = 0;
        }else if(gph->valid == 2){
            m_gsr->is_last_get = 1;
        }
		gph->valid = 1;
		gph->last_accessed_time = (unsigned)timestamp;

#ifdef DEBUG
		// DEBUG: Verify stored KEY matches
	    int ret = 0;
	    ret = d_memcmp((const void *)m_gph->key, (const void *)gph->key, m_gph->key_length);
	    if(ret != 1){
	        printf("KEYS NOT EQUAL!!\n");
	    }
#endif

		gpu_hash_lock[set_index] = UNLOCKED; // Unlock the set
	}

    __threadfence_system();
    /************************ End Critical Section ************************/

}


extern "C" __global__ void memcached_GET_kernel(unsigned long long first_req_addr,        // Address of first CUDA buffer containing a valid packet
                                                int num_req,                    // # of requests
                                                int *response_mem,              // Memory allocated for responses
                                                int hashpower,                  // Memcached hashpower
                                                unsigned int *gpu_hashtable,    // GPU resident Memcached hashtable
                                                int *gpu_hash_lock, // GPU resident locks for hashtable
                                                rel_time_t timestamp){

    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    int thread_type = ((local_tid % MAX_THREADS_PER_BLOCK) < (MAX_THREADS_PER_BLOCK / 2)) ? 0 : 1;// 0 means actual request threads, 1 means helper threads

    // This represents the request # that each thread will be responsible for. Request threads
    // will be from 0->NUM_REQUESTS_PER_GROUP
    // Each block handles 128 requests (minimum requests/batch), 256 threads per 128 requests.

    int group_id;

    if(local_tid < MAX_THREADS_PER_BLOCK){
    	group_id = 0;
    }else{
    	group_id = 1;
    }

    int half_group_size = MAX_THREADS_PER_BLOCK/2; // ==> 256/2 = 128

    int logical_tid = -1;
    if(thread_type == 0){
    	logical_tid = (group_id * half_group_size) + (tid % half_group_size); // First half looks
    }else{
    	logical_tid = (group_id * half_group_size) + ( (tid-half_group_size) % half_group_size);
    }


    _key_ m_key;	//	Local key per thread
    volatile gpu_primary_hashtable *g_primary_hashtable = (volatile gpu_primary_hashtable *)gpu_hashtable;	// Global Memcached Hash Table

	__shared__ mod_pkt_info mpi[NUM_REQUESTS_PER_GROUP];
	__shared__ int item_is_found[NUM_REQUESTS_PER_GROUP];


    m_key.key_len = 0;

    // Address of first packet. All other packets are pkt#*RX_BUFFER_SZ away from first_req_addr
    unsigned long long m_first_RX_buffer_addr = first_req_addr + (blockIdx.x * RX_BUFFER_SZ * NUM_REQUESTS_PER_GROUP);


    mod_parse_pkt(m_first_RX_buffer_addr, local_tid, logical_tid, thread_type, mpi, &m_key);

    __syncthreads();
    __threadfence();

#ifdef DEBUG
    if(mpi[logical_tid].is_get_req){
        printf("GET:::tid: %d, local_tid: %d, thread_type: %d, group_id: %d, logical_tid: %d\n", tid, local_tid, thread_type, group_id, logical_tid);
    }else{
        printf("FAIL:::tid: %d, local_tid: %d, thread_type: %d, group_id: %d, logical_tid: %d\n", tid, local_tid, thread_type, group_id, logical_tid);
    }
#endif

    if(mpi[logical_tid].is_get_req){
    	if(thread_type == 0){
    	    item_is_found[logical_tid] = mod_process_get_request(&mpi[logical_tid], hashpower, timestamp, g_primary_hashtable, &m_key, gpu_hash_lock);
		}else{
			mod_create_response_header(&mpi[logical_tid], logical_tid);
		}
	}

    __syncthreads();
    __threadfence();

    mod_populate_response((size_t *)response_mem, mpi, local_tid, logical_tid, group_id, item_is_found, thread_type, blockIdx.x, &m_key);

    __syncthreads();
    __threadfence_system();

}


// TODO: The coalesced packet load is currently hardcoded to 256 requests per sub-group 
//       and 512 threads per group. This likely doesn't need to change, but it could be 
//       made configurable. 
#define NUM_REQ_PER_LOOP	16
#define WARP_SIZE			32

#ifdef LATENCY_MEASURE
#define THREAD_PER_HDR_COPY	14	// 14 threads * 4 bytes = 56 bytes / hdr = 42 byte header + 8 byte memc hdr + "value "
#else
#define THREAD_PER_HDR_COPY	13	// 13 threads * 4 bytes = 52 bytes / hdr
#endif

__device__ void mod_parse_pkt(  unsigned long long first_RX_buffer_ptr,
                                int local_tid,
                                int logical_tid,
                                int thread_type,
                                mod_pkt_info *mpi,
                                _key_ *g_key){
	const char *GET = "get ";
	int *req_ptr = NULL;
	int *pkt_hdr_ptr = NULL;
	char *pkt_hdr = NULL;
    int ehs = sizeof(ether_header);
    int ips = sizeof(ip_header);
    int udps = sizeof(udp_header);
    unsigned network_size = ehs + ips + udps;
	ip_header *iph;
	udp_header *udp;
    char *payload;
    char *key;
    int count = 0;

    u_int16_t check = 0;

    int req_ind = (int)(local_tid / WARP_SIZE); // Which warp do you belong to?
    req_ind *= NUM_REQ_PER_LOOP;
	int w_tid = local_tid % WARP_SIZE;
	int masked_ind = w_tid % THREAD_PER_HDR_COPY;

	/**********************************************************/

    // Load packet headers from global to shared memory *coalesced accesses*
    // "LOAD PACKETS" stage from the SoCC paper.
    for(unsigned i=0; i<NUM_REQ_PER_LOOP; ++i){
        req_ptr = (int *)( first_RX_buffer_ptr  +  ((req_ind + i)*RX_BUFFER_SZ) );

        pkt_hdr_ptr = (int *)(&mpi[req_ind + i].nmch);
        pkt_hdr_ptr[masked_ind] = req_ptr[masked_ind];
    }

	__syncthreads();

    // "PARSE UDP PACKET" stage from the SoCC paper
	// The packet header contents are all in shared memory, now verify the packet contents (still in global mem)
    mpi[logical_tid].is_get_req = 1; // Assume all are UDP Memcached GET requests
    if(thread_type == 0){
        pkt_hdr = (char *)&mpi[logical_tid].nmch;
        iph = (ip_header *)(pkt_hdr + ehs);
        udp = (udp_header *)(pkt_hdr + ehs + ips);

        payload = (char *)(first_RX_buffer_ptr + (logical_tid*RX_BUFFER_SZ));

        payload += (network_size+8);

        if(G_NTOHS(udp->dest) != UDP_PORT){
            mpi[logical_tid].is_get_req = 0;
#ifdef DEBUG
            printf("UDP_PORT WRONG (%hu)\n", G_NTOHS(udp->dest));
#endif
        }

        // Verify Checksum
        // Lower 4-bits of version is the ip_header length (ihl)
        if(iph->check != 0){
            check = wrapsum(in_cksum((unsigned char *)iph, (iph->version & 0x0F)<<2, 0));
            if(check != 0){ 
                mpi[logical_tid].is_get_req = 0;
            }
        }

        if(mpi[logical_tid].is_get_req){
            for(unsigned i=0; i<3; ++i){
                if(payload[i] != GET[i]){
                    mpi[logical_tid].is_get_req = 0;
                }
            }
        }

        key = payload+4; // Move passed "get "
        if(mpi[logical_tid].is_get_req){
            // key is guaranteed to be a minimum of 16 bytes, load in 16 bytes as shorts.
            for(unsigned i=0; i<8; i++, count += 2){
                ((short *)(g_key->key))[i] = ((short *)(key))[i];
            }

            // Then load in the rest, searching for the end condition
            while( (key[count] != '\r') || (key[count+1] != '\n') ){
                g_key->key[count] = key[count];
                count++;
            }

            // Check if key is too large
            if(count >= MAX_KEY_SIZE){
                mpi[logical_tid].is_get_req = 0;
            }
        }

        // Set the key length
        g_key->key_len = count;
    }

}


// Actual Memcached hash + key lookup
// "Network Service Processing" stage in the SoCC paper
__device__ int mod_process_get_request(mod_pkt_info *mpi, int hashpower, rel_time_t time,
									volatile gpu_primary_hashtable *g_primary_hashtable,
									_key_ *g_key,
                                    int *gpu_hash_lock){
    unsigned hv;
    int ret = 0;

    size_t nkey = g_key->key_len;
    char *key = g_key->key;
    volatile char *key_t;
    unsigned key_hash_t;
    volatile gpu_primary_hashtable *m_gph;

    int is_locked = 0;

    volatile int old_lock_val = -1;
    volatile int new_lock_val = 0;
    volatile int new_old_lock_val = 0;

    unsigned set_index;
    unsigned set_hv_index;

    unsigned key_hash = 0;

    // Compute the hash
    hv = hash(key, nkey, 0);
    key_hash = hv & KEY_HASH_MASK; // Compute the hash mask for this key

    // Compute the set index for the hash and the corresponding index into the hash table
    unsigned num_sets = hashsize(hashpower) / SET_ASSOC_SIZE;
    set_index = hv % num_sets;                  // Calculate the set index for this hash value
    set_hv_index = set_index*SET_ASSOC_SIZE;    // Move to the correct location in the hash table for this set


    // Soft mutex for each GET request. Multiple shared_locks, only single private_lock.
    // Grab the shared lock for the set
    while(!is_locked){
        old_lock_val = gpu_hash_lock[set_index];
        if(old_lock_val != -1){ // TEST
            new_lock_val = old_lock_val+1;
            new_old_lock_val = atomicCAS(&gpu_hash_lock[set_index], old_lock_val, new_lock_val); // and TEST and SET
            if(new_old_lock_val == old_lock_val){
                is_locked = 1;
            }
        }
    }

    // Set initial response length if item isn't found
    mpi->pkt_length = RESPONSE_SIZE;
    
    /************************ Critical Section ************************/
    for(unsigned i=0; i<SET_ASSOC_SIZE; ++i){
        m_gph = (volatile gpu_primary_hashtable *)&g_primary_hashtable[set_hv_index + i];

        if(m_gph->valid > 0){
            key_t = (volatile char *)m_gph->key;

            // New - First check key hash. If equal, then do key comparison. Otherwise, no way they're equal.
            key_hash_t = m_gph->key_hash;
            if(key_hash == key_hash_t){
                ret = fast_memcmp((const void *)key, (const void *)key_t, nkey);

                if(ret){
                    mpi->it = (item *)m_gph->item_ptr;      // Update response pointer
#ifndef CONSTANT_RESPONSE_SIZE
                    mpi->pkt_length = m_gph->pkt_length;    // Update value length for response packet size
#endif
                    m_gph->last_accessed_time = time; // Possible Race Condition if multiple GETs updating this concurrently, but don't care who wins                    
                    m_gph->valid = 2; // Update hash table entry to say that last access was a GET request
                    break;
                }
            }
        }
    }

    // Unlock the set
    atomicSub(&gpu_hash_lock[set_index], 1);
    /************************ End Critical Section ************************/

    return ret;
}



__device__ void mod_create_response_header(mod_pkt_info *mpi, int helper_tid){
	// m_res points to correct response memory for this helper_thread
	// mpi contains unmodified packet header, modify in shared memory

	// Elements to swap
	u_int8_t  ether_swap;
	u_int16_t ip_addr1;
	u_int16_t ip_addr2;
	u_int16_t udp_port;

	const char *VALUE = "VALUE ";

	char *header = (char *)(&mpi->nmch);

	ether_header *eh = (ether_header *)header;
	ip_header *iph = (ip_header *)&header[14];
	udp_header *uh = (udp_header *)&header[34];

	// Swap ether
	for(unsigned i=0; i<ETH_ALEN; ++i){
		ether_swap = eh->ether_shost[i];
		eh->ether_shost[i] = eh->ether_dhost[i];
		eh->ether_dhost[i] = ether_swap;
	}

	// Swap IP
	ip_addr1 = iph->saddr1;
	ip_addr2 = iph->saddr2;
	iph->saddr1 = iph->daddr1;
	iph->saddr2 = iph->daddr2;
	iph->daddr1 = ip_addr1;
	iph->daddr2 = ip_addr2;
	iph->check = 0;

	// Swap UDP port
	udp_port = uh->source;
	uh->source = uh->dest;
	uh->dest = udp_port;
    uh->check = 0;

#ifdef CONSTANT_RESPONSE_SIZE
	// Assume a constant response packet and calculate the checksum
	// (used to force response packets to be smaller than the request
	// packets so we can reach peak 13 MRPS throughput with 16 Byte keys).
	iph->tot_len = G_HTONS((RESPONSE_SIZE - sizeof(ether_header)));
	uh->len = G_HTONS((RESPONSE_SIZE - sizeof(ether_header) - sizeof(ip_header)));
	iph->check = wrapsum(in_cksum((unsigned char *)iph, 4*(iph->version & 0x0F), 0));
#else

	// Calculate an initial partial checksum without the IP header length field.
	// This will be added in afterwards
	iph->check = partial_cksum((unsigned char *)iph, 4*(iph->version & 0x0F), 0);

#endif

	// Copy in "VALUE "
#ifdef LATENCY_MEASURE
    // If doing latency measurements, add the 8byte memc header before "VALUE " (Memc hdr used for a client timestamp)
	mpi->nmch.mch.hdr[8] = VALUE[0];
	mpi->nmch.mch.hdr[9] = VALUE[1];
	mpi->nmch.mch.hdr[10] = VALUE[2];
	mpi->nmch.mch.hdr[11] = VALUE[3];
	mpi->nmch.mch.hdr[12] = VALUE[4];
	mpi->nmch.mch.hdr[13] = VALUE[5];

#else
	mpi->nmch.mch.hdr[0] = VALUE[0];
	mpi->nmch.mch.hdr[1] = VALUE[1];
	mpi->nmch.mch.hdr[2] = VALUE[2];
	mpi->nmch.mch.hdr[3] = VALUE[3];
	mpi->nmch.mch.hdr[4] = VALUE[4];
	mpi->nmch.mch.hdr[5] = VALUE[5];

#endif

	return;
}


__device__ void mod_populate_response(size_t *res_mem, mod_pkt_info *mpi, int local_tid, int logical_tid, int group_id, int *item_is_found, unsigned thread_type, int cta_id, _key_ *g_key){

	int *res_ptr = NULL;
	int *pkt_hdr_ptr = NULL;

	int item_ptr_ind = cta_id*NUM_REQUESTS_PER_GROUP + logical_tid;

    int req_ind = (int)(local_tid / WARP_SIZE); // Which warp this thread belongs to
    req_ind *= NUM_REQ_PER_LOOP;
	int w_tid = local_tid % WARP_SIZE;
	int masked_ind = w_tid % THREAD_PER_HDR_COPY;

    mod_pkt_info *m_mpi = &mpi[logical_tid];

	pkt_res_memc_hdr *start_response_pkt_hdr_mem = (pkt_res_memc_hdr *)(res_mem + NUM_REQUESTS_PER_BATCH);
    pkt_res_memc_hdr *response_pkt_hdr_mem = (pkt_res_memc_hdr *)&start_response_pkt_hdr_mem[cta_id*NUM_REQUESTS_PER_GROUP];


	if(thread_type == 0){ // Thread_type 0 stores the found item pointers
	    //res_mem[item_ptr_ind] = (size_t)NULL;
    	if(item_is_found[logical_tid]){
    	    res_mem[item_ptr_ind] = (size_t)mpi[logical_tid].it;
	    }else{
            res_mem[item_ptr_ind] = (size_t)NULL;
        }
    }
#ifndef CONSTANT_RESPONSE_SIZE // If not using a constant response size, set the packet length fields and update checksum
	else {
	    char *header = (char *)(&m_mpi->nmch);
	    ip_header *iph = (ip_header *)&header[14];
	    udp_header *uh = (udp_header *)&header[34];

	    // Update response packet lengths and compute IP checksum
	    iph->tot_len = G_HTONS((m_mpi->pkt_length - sizeof(ether_header)));
	    uh->len = G_HTONS((m_mpi->pkt_length - sizeof(ether_header) - sizeof(ip_header)));

	    // Already computed a partial checksum without the IP header length field.
        // Add the updated length to the checksum. 
        iph->check = wrapsum(cksum_hdr_len_only((unsigned char *)iph, iph->check));

	}
#endif

	__syncthreads();

    // Finally, store packet response headers from shared to global memory
	for(unsigned i=0; i<NUM_REQ_PER_LOOP; ++i){
		pkt_hdr_ptr = (int *)(&mpi[req_ind + i].nmch);
		res_ptr = (int *)&response_pkt_hdr_mem[req_ind + i];
		res_ptr[masked_ind] = pkt_hdr_ptr[masked_ind]; // This copies over the pkt hdr + "VALUE "
	}

	__syncthreads();

}
