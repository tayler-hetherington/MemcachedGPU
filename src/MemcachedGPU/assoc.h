/* associative array */
void assoc_init(CUDAContext *cuda_context, const int hashpower_init);
item *assoc_find(const char *key, const size_t nkey, const uint32_t hv);

// GNoM - Tayler
int gnom_assoc_insert(item *item, const uint32_t hv);

int assoc_insert(item *item, const uint32_t hv);
void assoc_delete(const char *key, const size_t nkey, const uint32_t hv);
void do_assoc_move_next_bucket(void);
int start_assoc_maintenance_thread(void);
void stop_assoc_maintenance_thread(void);
extern unsigned int hashpower;
unsigned int get_hashtable_size();
unsigned int get_hashlock_size();
