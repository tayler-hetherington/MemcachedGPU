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
 * gpu_common.c
 */

#include <sys/time.h>
#include <unistd.h>

#include "gpu_common.h"

double TSC_MULT=0;
int timer_init=0;

/*
__inline__ uint64_t RDTSC(void) {
	uint32_t lo, hi;
	__asm__ __volatile__ (
	"		xorl %%eax, %%eax \n"
	"		cpuid"
	::: "%rax", "%rbx", "%rcx", "rdx");
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return (uint64_t)hi << 32 | lo;
}
*/

void init_timer(){
	long long TscStart, TscEnd;
	double TscTotal = 0;
	struct timeval tv;

	printf("Calibrating...\n");

	gettimeofday(&tv, NULL);
	double start = (double) tv.tv_sec + (tv.tv_usec/1000000.0);
	TscStart = RDTSC();
	sleep(1);
	TscEnd = RDTSC();
	gettimeofday(&tv, NULL);
	double end = (double) tv.tv_sec + (tv.tv_usec/1000000.0);
	double diff = end-start;

	TscTotal += (double)(TscEnd - TscStart);
	TSC_MULT = (unsigned)((double)(TscTotal / diff) + 1.0);

	printf("Calc complete... TSC: %lf	TOD: %lf	MULT: %lf\n", TscTotal/TSC_MULT, diff, TSC_MULT);

	printf("Verifying result, sleeping for 5 msec...\n");
	TscTotal = 0;
	TscStart = RDTSC();
	usleep(5000);
	TscEnd = RDTSC();
	TscTotal += (double)(TscEnd - TscStart);
	printf("TSC Result: %f, end-start: %f\n", TscTotal/TSC_MULT, TscTotal);
	timer_init = 1;
}

