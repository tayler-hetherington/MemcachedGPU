/* -*- Mode: C; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/*
 * Hash table
 *
 * The hash function used here is by Bob Jenkins, 1996:
 *    <http://burtleburtle.net/bob/hash/doobs.html>
 *       "By Bob Jenkins, 1996.  bob_jenkins@burtleburtle.net.
 *       You may use this code any way you wish, private, educational,
 *       or commercial.  It's free."
 *
 * The rest of the file is licensed under the BSD license.  See LICENSE.
 */

/**********************************************************************
Modified by Tayler Hetherington 2015
The University of British Columbia
***********************************************************************/

#include "memcached.h"
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>


static pthread_cond_t maintenance_cond = PTHREAD_COND_INITIALIZER;


typedef  unsigned long  int  ub4;   /* unsigned 4-byte quantities */
typedef  unsigned       char ub1;   /* unsigned 1-byte quantities */

/* how many powers of 2's worth of buckets we use */
unsigned int hashpower = HASHPOWER_DEFAULT;

#define hashsize(n) ((ub4)1<<(n))
#define hashmask(n) (hashsize(n)-1)

/* Main hash table. This is where we look except during expansion. */
item** primary_hashtable = 0;

/*
 * Previous hash table. During expansion, we look here for keys that haven't
 * been moved over to the primary yet.
 */
item** old_hashtable = 0;

/* Number of items in the hash table. */
static unsigned int hash_items = 0;

/* Flag: Are we in the middle of expanding now? */
bool expanding = false;
bool started_expanding = false;

/*
 * During expansion we migrate values with bucket granularity; this is how
 * far we've gotten so far. Ranges from 0 .. hashsize(hashpower - 1) - 1.
 */
unsigned int expand_bucket = 0;

unsigned int get_hashtable_size(){
	return hashsize(hashpower)*sizeof(gpu_primary_hashtable);
}

unsigned int get_hashlock_size(){
#ifdef SET_ASSOC_SIZE
    return (hashsize(hashpower) / SET_ASSOC_SIZE)*sizeof(int);
#else
    return hashsize(hashpower)*sizeof(int);
#endif
}

void assoc_init(CUDAContext *cuda_context, const int hashtable_init) {
    if (hashtable_init) {
        hashpower = hashtable_init;
    }
    
    primary_hashtable = calloc(hashsize(hashpower), sizeof(void *));
    if (! primary_hashtable) {
        fprintf(stderr, "Failed to init hashtable.\n");
        exit(EXIT_FAILURE);
    }
    STATS_LOCK();
    stats.hash_power_level = hashpower;
    stats.hash_bytes = hashsize(hashpower) * sizeof(void *);
    STATS_UNLOCK();
}

item *assoc_find(const char *key, const size_t nkey, const uint32_t hv) {
    item *it;
    unsigned int oldbucket;

    if (expanding &&
        (oldbucket = (hv & hashmask(hashpower - 1))) >= expand_bucket)
    {
        it = old_hashtable[oldbucket];
    } else {
        it = primary_hashtable[hv & hashmask(hashpower)];
    }

    item *ret = NULL;
    int depth = 0;
    while (it) {
        if ((nkey == it->nkey) && (memcmp(key, ITEM_key(it), nkey) == 0)) {
            ret = it;
            break;
        }
        it = it->h_next;
        ++depth;
    }
    MEMCACHED_ASSOC_FIND(key, nkey, depth);
    return ret;
}

/* returns the address of the item pointer before the key.  if *item == 0,
   the item wasn't found */

static item** _hashitem_before (const char *key, const size_t nkey, const uint32_t hv) {
    item **pos;
    unsigned int oldbucket;

    if (expanding &&
        (oldbucket = (hv & hashmask(hashpower - 1))) >= expand_bucket)
    {
        pos = &old_hashtable[oldbucket];
    } else {
        pos = &primary_hashtable[hv & hashmask(hashpower)];
    }

    while (*pos && ((nkey != (*pos)->nkey) || memcmp(key, ITEM_key(*pos), nkey))) {
        pos = &(*pos)->h_next;
    }
    return pos;
}

int gnom_assoc_insert(item *it, const uint32_t hv) {

    // GNoM: Run CUDA SET kernel to insert the item
    // This may trigger an eviction on a hash table set-associative set collision OR 
    // on a SET hit. To preserver consistency of concurrent GETs and SETs without locks, 
    // SETs must preserve GPU ordering. 
    int req_size = 2*sizeof(gpu_set_req);
    int res_size = sizeof(gpu_set_res);
    rel_time_t timestamp = current_time;

    gpu_set_req *req_mem = (gpu_set_req *)calloc(req_size, 1);
    gpu_set_res *res_mem = (gpu_set_res *)calloc(res_size, 1);

    // TODO: Currently testing with a maximum key size of 128. This can change to anything but
    //       need to update the max key length in the CUDA kernels as well.
    assert(it->nkey <= 128);
	memcpy(req_mem->key, ITEM_key(it), it->nkey); // Copy the key

	req_mem->init_hv = hv;
	req_mem->item_ptr = (void *)it;
	req_mem->key_length = it->nkey; // Set key length

	// Total packet size = Network header + "VALUE " + key + suffix + data (with "\r\n")
	req_mem->pkt_length = NETWORK_HDR_SIZE + 6 + it->nkey + it->nsuffix + it->nbytes;
	it->gnom_total_response_length = req_mem->pkt_length;

    // Grab the GNoM SET lock - Currently serialize all SETs
    gnom_set_lock();

#ifdef GNOM_SET_DEBUG
        printf("GNoM: GNoM SET request populated, launching on GPU: item=%p, len=%d, key=%s\n", req_mem->item_ptr, req_mem->key_length, req_mem->key);
#endif

    // GNoM: Send SET request to the GNoM flow
	send_set_stream_request((void *)req_mem, req_size, (void *)res_mem, res_size, timestamp);

    if(res_mem->is_evicted > 0){
        // Need to evict something from the hash table
        if(res_mem->is_evicted == 1){ // If need to evict to fit space in the hash table
            inc_set_evict();
#ifdef GNOM_SET_DEBUG
            printf("\tEvict item: Hash table set collision\n");
#endif
        }else if(res_mem->is_evicted == 2){  // If the SET/UPDATE hit
            inc_set_hit();
#ifdef GNOM_SET_DEBUG          
            printf("\tEvict item: Set hit\n");
#endif
        }

        // Stall here until any potential conflicting and concurrent GET requests have completed 
        // Ordering is enforced by the exclusive GPU locks 
        // Other option - check that the evicted item was last touched by a SET. If it was a SET, then ordering is enfored by the SET locks already
        //    no need to wait for GET batches to complete
        if(res_mem->is_last_get == 0 ||  gnom_poll_get_complete_timestamp(res_mem->evicted_lru_timestamp)){
            // Now we ensure that any GET request that was looking at the evicted item is now 
            // complete and already accessed the item. The hash table was updated by this SET, 
            // so no other GET requests should see this item. It's free to remove.

#ifdef GNOM_SET_DEBUG
            printf("GNoM: All potentially conflicting GETs complete, removing item: %p\n", res_mem->evicted_ptr);
#endif
            gnom_item_remove((item *)res_mem->evicted_ptr);  

        }else{
            printf("GNoM Error: Shouldn't ever be here\n");
            abort();
        }
    }else{
        inc_set_miss(); // If no eviction is required, then this must be a miss
#ifdef GNOM_SET_DEBUG
            printf("GNoM: No evict, item succesfully stored\n");
#endif
    }

    MEMCACHED_ASSOC_INSERT(ITEM_key(it), it->nkey, hash_items);

#ifdef GNOM_SET_DEBUG
            printf("GNoM: SET on GPU complete!\n");
#endif

    // Release the GNoM SET lock
    gnom_set_unlock();

    return 1;
}




/* grows the hashtable to the next power of 2. */
static void assoc_expand(void) {

    old_hashtable = primary_hashtable;
    primary_hashtable = calloc(hashsize(hashpower + 1), sizeof(void *));

    if (primary_hashtable) {
        if (settings.verbose > 1)
            fprintf(stderr, "Hash table expansion starting\n");
        hashpower++;
        expanding = true;
        expand_bucket = 0;
        STATS_LOCK();
        stats.hash_power_level = hashpower;
        stats.hash_bytes += hashsize(hashpower) * sizeof(void *);
        stats.hash_is_expanding = 1;
        STATS_UNLOCK();
    } else {
        primary_hashtable = old_hashtable;
        /* Bad news, but we can keep running. */
    }
}

static void assoc_start_expand(void) {
    if (started_expanding)
        return;
    started_expanding = true;
    pthread_cond_signal(&maintenance_cond);
}

/* Note: this isn't an assoc_update.  The key must not already exist to call this */
int assoc_insert(item *it, const uint32_t hv) {
    unsigned int oldbucket;

    if (expanding &&
        (oldbucket = (hv & hashmask(hashpower - 1))) >= expand_bucket)
    {
        it->h_next = old_hashtable[oldbucket];
        old_hashtable[oldbucket] = it;
    } else {
        it->h_next = primary_hashtable[hv & hashmask(hashpower)];
        primary_hashtable[hv & hashmask(hashpower)] = it;
    }

    hash_items++;
    if (! expanding && hash_items > (hashsize(hashpower) * 3) / 2) {
        assoc_start_expand();
    }

    MEMCACHED_ASSOC_INSERT(ITEM_key(it), it->nkey, hash_items);
    return 1;
}

void assoc_delete(const char *key, const size_t nkey, const uint32_t hv) {
    item **before = _hashitem_before(key, nkey, hv);

    if (*before) {
        item *nxt;
        hash_items--;
        /* The DTrace probe cannot be triggered as the last instruction
         * due to possible tail-optimization by the compiler
         */
        MEMCACHED_ASSOC_DELETE(key, nkey, hash_items);
        nxt = (*before)->h_next;
        (*before)->h_next = 0;   /* probably pointless, but whatever. */
        *before = nxt;
        return;
    }
    /* Note:  we never actually get here.  the callers don't delete things
       they can't find. */
    assert(*before != 0);
}


static volatile int do_run_maintenance_thread = 1;

#define DEFAULT_HASH_BULK_MOVE 1
int hash_bulk_move = DEFAULT_HASH_BULK_MOVE;

static void *assoc_maintenance_thread(void *arg) {

    while (do_run_maintenance_thread) {
        int ii = 0;

        /* Lock the cache, and bulk move multiple buckets to the new
         * hash table. */
        item_lock_global();
        mutex_lock(&cache_lock);

        for (ii = 0; ii < hash_bulk_move && expanding; ++ii) {
            item *it, *next;
            int bucket;

            for (it = old_hashtable[expand_bucket]; NULL != it; it = next) {
                next = it->h_next;

                bucket = hash(ITEM_key(it), it->nkey, 0) & hashmask(hashpower);
                it->h_next = primary_hashtable[bucket];
                primary_hashtable[bucket] = it;
            }

            old_hashtable[expand_bucket] = NULL;

            expand_bucket++;
            if (expand_bucket == hashsize(hashpower - 1)) {
                expanding = false;                

                free(old_hashtable);

                STATS_LOCK();
                stats.hash_bytes -= hashsize(hashpower - 1) * sizeof(void *);
                stats.hash_is_expanding = 0;
                STATS_UNLOCK();
                if (settings.verbose > 1)
                    fprintf(stderr, "Hash table expansion done\n");
            }
        }

        mutex_unlock(&cache_lock);
        item_unlock_global();

        if (!expanding) {
            /* finished expanding. tell all threads to use fine-grained locks */
            switch_item_lock_type(ITEM_LOCK_GRANULAR);
            slabs_rebalancer_resume();
            /* We are done expanding.. just wait for next invocation */
            mutex_lock(&cache_lock);
            started_expanding = false;
            pthread_cond_wait(&maintenance_cond, &cache_lock);
            /* Before doing anything, tell threads to use a global lock */
            mutex_unlock(&cache_lock);
            slabs_rebalancer_pause();
            switch_item_lock_type(ITEM_LOCK_GLOBAL);
            mutex_lock(&cache_lock);
            assoc_expand();
            mutex_unlock(&cache_lock);
        }
    }
    return NULL;
}

static pthread_t maintenance_tid;

int start_assoc_maintenance_thread() {
    int ret;
    char *env = getenv("MEMCACHED_HASH_BULK_MOVE");
    if (env != NULL) {
        hash_bulk_move = atoi(env);
        if (hash_bulk_move == 0) {
            hash_bulk_move = DEFAULT_HASH_BULK_MOVE;
        }
    }
    if ((ret = pthread_create(&maintenance_tid, NULL,
                              assoc_maintenance_thread, NULL)) != 0) {
        fprintf(stderr, "Can't create thread: %s\n", strerror(ret));
        return -1;
    }
    return 0;
}

void stop_assoc_maintenance_thread() {
    mutex_lock(&cache_lock);
    do_run_maintenance_thread = 0;
    pthread_cond_signal(&maintenance_cond);
    mutex_unlock(&cache_lock);

    /* Wait for the maintenance thread to stop */
    pthread_join(maintenance_tid, NULL);
}


