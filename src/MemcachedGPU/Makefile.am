bin_PROGRAMS = memcached
pkginclude_HEADERS = protocol_binary.h
noinst_PROGRAMS = memcached-debug sizes testapp timedrun

BUILT_SOURCES=

testapp_SOURCES = testapp.c util.c util.h

timedrun_SOURCES = timedrun.c

memcached_SOURCES = memcached.c memcached.h \
                    hash.c hash.h \
                    slabs.c slabs.h \
                    items.c items.h \
                    assoc.c assoc.h \
                    thread.c daemon.c \
                    stats.c stats.h \
                    util.c util.h \
                    trace.h cache.h sasl_defs.h \
					gpu_common.c gpu_common.h \
					cuda_stream_wrapper.cpp cuda_stream_wrapper.h \
					cuda_memory_manager.cpp cuda_memory_manager.h \
					cuda_stream_manager.cpp cuda_stream_manager.h \
					cuda_context_manager.cpp cuda_context_manager.h

memcached_debug_SOURCES = $(memcached_SOURCES)

if BUILD_CACHE
memcached_SOURCES += cache.c
testapp_SOURCES += cache.c
endif

if BUILD_SOLARIS_PRIVS
memcached_SOURCES += solaris_priv.c
endif

if ENABLE_SASL
memcached_SOURCES += sasl_defs.c
endif

# 
# PF_RING specific pointers
#
PFRINGDIR = $(PF_RING_DIR)/userland/lib
LIBPFRING = $(PFRINGDIR)/libpfring.a
PCAPDIR = $(PF_RING_DIR)/userland/libpcap-1.1.1-ring
LIBPCAP = $(PCAPDIR)/libpcap.a
PFRING_KERNEL = $(PF_RING_DIR)/kernel
PFRING_INCLUDE = -I$(PFRINGDIR) -I$(PFRING_KERNEL)
PFRING_LIBS = $(LIBPFRING) $(LIBPCAP) -lrt -lnuma -march=corei7-avx -mtune=corei7-avx

LIBS += $(PFRING_LIBS) -lcuda -lcudart -lpthread

GNOM_KM_DIR ?= ../GNoM_km/
 
memcached_CPPFLAGS = -DNDEBUG -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc/ $(PFRING_INCLUDE)
memcached_LDADD = @PROFILER_LDFLAGS@ -L/usr/lib/ -L$(CUDA_INSTALL_DIR)/lib64/ -L$(CUDA_INSTALL_DIR)/lib/ -L$(CUDA_COMMON_DIR)/lib/ -L$(CUDA_SHARED)/lib/ 
memcached_CFLAGS = @PROFILER_FLAGS@ -g -DNDEBUG -g -pthread -Wall -Wredundant-decls -I$(CUDA_INSTALL_DIR)/include -I$(CUDA_COMMON_DIR)/inc/ $(PFRING_INCLUDE)

memcached_debug_CPPFLAGS = -g -DNDEBUG -g -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc/ $(PFRING_INCLUDE)
memcached_debug_LDADD = @PROFILER_LDFLAGS@ -L/usr/lib/ -L$(CUDA_INSTALL_DIR)/lib64/ -L$(CUDA_INSTALL_DIR)/lib/ -L$(CUDA_COMMON_DIR)/lib/ 
memcached_debug_CFLAGS = @PROFILER_FLAGS@ -g -DNDEBUG -g -pthread -Wall -Wredundant-decls -I$(CUDA_INSTALL_DIR)/include -I$(CUDA_COMMON_DIR)/inc/ $(PFRING_INCLUDE)


AM_CFLAGS = -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc/ -I$(GNOM_KM_DIR) $(PFRING_INCLUDE)
AM_CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include/ -I$(CUDA_COMMON_DIR)/inc/ -I$(GNOM_KM_DIR) $(PFRING_INCLUDE)
AM_LDFLAGS = -L/usr/lib/ -L$(CUDA_INSTALL_DIR)/lib64/ -L$(CUDA_INSTALL_DIR)/lib/ -L$(CUDA_COMMON_DIR)/lib/ -L$(GNOM_KM_DIR)


memcached_DEPENDENCIES =
memcached_debug_DEPENDENCIES =
CLEANFILES=

if BUILD_DTRACE
BUILT_SOURCES += memcached_dtrace.h
CLEANFILES += memcached_dtrace.h
endif

if DTRACE_INSTRUMENT_OBJ
memcached_LDADD += memcached_dtrace.o
memcached_DEPENDENCIES += memcached_dtrace.o
memcached_debug_LDADD += memcached_debug_dtrace.o
memcached_debug_DEPENDENCIES += memcached_debug_dtrace.o
CLEANFILES += memcached_dtrace.o memcached_debug_dtrace.o
endif

memcached_dtrace.h: memcached_dtrace.d
	${DTRACE} -h -s memcached_dtrace.d
	sed -e 's,void \*,const void \*,g' memcached_dtrace.h | \
            sed -e 's,char \*,const char \*,g' | tr '\t' ' ' > mmc_dtrace.tmp
	mv mmc_dtrace.tmp memcached_dtrace.h

memcached_dtrace.o: $(memcached_OBJECTS)
	$(DTRACE) $(DTRACEFLAGS) -G -o memcached_dtrace.o -s ${srcdir}/memcached_dtrace.d $(memcached_OBJECTS)

memcached_debug_dtrace.o: $(memcached_debug_OBJECTS)
	$(DTRACE) $(DTRACEFLAGS) -G -o memcached_debug_dtrace.o -s ${srcdir}/memcached_dtrace.d $(memcached_debug_OBJECTS)


SUBDIRS = doc
DIST_DIRS = scripts
EXTRA_DIST = doc scripts t memcached.spec memcached_dtrace.d version.m4 README.md

MOSTLYCLEANFILES = *.gcov *.gcno *.gcda *.tcov

test:	memcached-debug sizes testapp
	$(srcdir)/sizes
	$(srcdir)/testapp
	prove $(srcdir)/t
	@if test `basename $(PROFILER)` = "gcov"; then \
	  for file in memcached_debug-*.gc??; do \
	    mv -f $$file `echo $$file | sed 's/memcached_debug-//'`; \
	  done && \
	  for file in *.gcda; do \
	    srcfile=`echo $$file | sed 's/.gcda/.c/'`; \
	    if test -n "`echo $(memcached_debug_SOURCES) | grep $$srcfile`"; then \
	      echo `$(PROFILER) $$srcfile` | sed 's/'$$srcfile':.*//'; \
	    fi \
	  done \
	elif test `basename $(PROFILER)` = "tcov"; then \
	  files=`grep SRCFILE memcached-debug.profile/tcovd | sed 's/SRCFILE://' | sort | uniq` && \
	  $(PROFILER) -x memcached-debug.profile $$files 2>&1; \
	  for file in *.tcov; do \
	    srcfile=`echo $$file | sed 's/.tcov//'`; \
	    if test -n "`echo $(memcached_debug_SOURCES) | grep $$srcfile`"; then \
	      echo $$srcfile : `grep 'Percent of the file executed' $$file`; \
	    fi \
	  done \
	else :; fi

docs:
	(cat Doxyfile ; echo "PROJECT_NUMBER=`cat version.num`") | doxygen -

dist-hook:
	rm -f $(distdir)/*/*~ $(distdir)/t/lib/*~ $(distdir)/*~
