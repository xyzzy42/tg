/*
    tg
    Copyright (C) 2022 Trent Piepho

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 2 as
    published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

/* Fast implementation of a True Peak Programme Meter
 *
 * Based on ITU-R BS.1770-4, Annex 2, "Guidelines for accurate measurement of "true-peak" level".
 *
 * The goal is to the find the true peak level of the signal.  If we only find the peak sample
 * then the true peak might be missed if it lies between samples.  The algorithm used is to 4x
 * over-sample the data and then apply a low pass filter.  Effectively, this interpolates
 * between each sample with an estimation of the true waveform.
 *
 * BS.1770 provides a 48 tap 4 phase FIR filter that meets the standard's low pass filter
 * requirements.  However, it does not specify what those requirements are.  The provided filter
 * appears to have been created by Remez exchange.  The following specification produces nearly
 * identical coefficients.
 *
 * Pass band: 0 - (19/192)*Fs
 * gain = 4
 * desired ripple = 0.2 dB
 *
 * Stop Band: (29/192)*Fs - ½Fs
 * gain = 0
 * desired attenuation = -27.809 dB
 *
 * To improve performance, a shorter 36 tap 4 phase filter is used with the same parameters for
 * the Remez algorithm.  This results in an actual pass-band ripple of 0.2053527060589797 dB and
 * stop-bad attenuation of -24.951520064797474 dB.
 */

#include "tg.h"
#include <assert.h>

/* Filter size, 36 taps in 4 phases. */
#define FILTER_LEN	9
#define PHASES		4
/* Coefficients of 36 tap low pass filter. */
#define D00  0.0347432144544397026
#define D01  0.0053033398849038956
#define D02 -0.0158056839910024130
#define D03 -0.0401345218720167846
#define D04 -0.0487592254838851574
#define D05 -0.0271657261314119207
#define D06  0.0217773339266858663
#define D07  0.0731537021832674994
#define D08  0.0902070324455883366
#define D09  0.0465427351307140594
#define D10 -0.0514525618099469506
#define D11 -0.1562738114671321787
#define D12 -0.1952349715419243459
#define D13 -0.1052304280525301616
#define D14  0.1295519332069564600
#define D15  0.4588520584626076815
#define D16  0.7794020794555769349
#define D17  0.9768824983763817471
#define D18  0.9768824983763817471
#define D19  0.7794020794555769349
#define D20  0.4588520584626076815
#define D21  0.1295519332069564600
#define D22 -0.1052304280525301616
#define D23 -0.1952349715419243459
#define D24 -0.1562738114671321787
#define D25 -0.0514525618099469506
#define D26  0.0465427351307140594
#define D27  0.0902070324455883366
#define D28  0.0731537021832674994
#define D29  0.0217773339266858663
#define D30 -0.0271657261314119207
#define D31 -0.0487592254838851574
#define D32 -0.0401345218720167846
#define D33 -0.0158056839910024130
#define D34  0.0053033398849038956
#define D35  0.0347432144544397026

/** A loop over each filter phase. */
#define for_each_phase(v) for (unsigned v = 0; v < PHASES; v++)

/** For alignment of SIMD data */
#define align(x) __attribute__((aligned(x)))

#if __AVX2__ && __FMA__
#include <immintrin.h>

/** AVX2 implementation of finding the maximum of a polyphase FIR filter output.
 *
 * This operates on 256-bit AVX registers, which contain eight 32-bit floats each.  Two
 * registers are loaded with 16 input samples per iteration.  A series of shifts/blends/permutes
 * on these two regisers produces the sequence of samples 0-8 in the first slot of the registers,
 * 1-9 in the next slot, and so on to 7-15 in the last slot.  But, the samples are not produced
 * strictly in sequential order.  This is accounted for in the table of filter coefficients.
 *
 * The result is a register with 8 filtered output data values for one phase.  Four phases are
 * done in parallel with the same input data for each phase but a different set of coefficients.
 * Thus 32 output values are produced each iteration.
 *
 * The maximum absolute value of the filter output is returned.
 */
static float findtppm_avx(const float* x, unsigned len)
{
	static const __m256 align(32) c[FILTER_LEN][PHASES] = {
		{ {D00, D00, D00, D00, D00, D00, D00, D00},
		  {D01, D01, D01, D01, D01, D01, D01, D01},
		  {D02, D02, D02, D02, D02, D02, D02, D02},
		  {D03, D03, D03, D03, D03, D03, D03, D03} },
		{ {D04, D04, D04, D20, D04, D04, D04, D20},
		  {D05, D05, D05, D21, D05, D05, D05, D21},
		  {D06, D06, D06, D22, D06, D06, D06, D22},
		  {D07, D07, D07, D23, D07, D07, D07, D23} },
		{ {D08, D08, D24, D24, D08, D08, D24, D24},
		  {D09, D09, D25, D25, D09, D09, D25, D25},
		  {D10, D10, D26, D26, D10, D10, D26, D26},
		  {D11, D11, D27, D27, D11, D11, D27, D27} },
		{ {D12, D28, D28, D28, D12, D28, D28, D28},
		  {D13, D29, D29, D29, D13, D29, D29, D29},
		  {D14, D30, D30, D30, D14, D30, D30, D30},
		  {D15, D31, D31, D31, D15, D31, D31, D31} },
		{ {D16, D16, D16, D16, D16, D16, D16, D16},
		  {D17, D17, D17, D17, D17, D17, D17, D17},
		  {D18, D18, D18, D18, D18, D18, D18, D18},
		  {D19, D19, D19, D19, D19, D19, D19, D19} },
		{ {D20, D20, D20, D04, D20, D20, D20, D04},
		  {D21, D21, D21, D05, D21, D21, D21, D05},
		  {D22, D22, D22, D06, D22, D22, D22, D06},
		  {D23, D23, D23, D07, D23, D23, D23, D07} },
		{ {D24, D24, D08, D08, D24, D24, D08, D08},
		  {D25, D25, D09, D09, D25, D25, D09, D09},
		  {D26, D26, D10, D10, D26, D26, D10, D10},
		  {D27, D27, D11, D11, D27, D27, D11, D11} },
		{ {D28, D12, D12, D12, D28, D12, D12, D12},
		  {D29, D13, D13, D13, D29, D13, D13, D13},
		  {D30, D14, D14, D14, D30, D14, D14, D14},
		  {D31, D15, D15, D15, D31, D15, D15, D15} },
		{ {D32, D32, D32, D32, D32, D32, D32, D32},
		  {D33, D33, D33, D33, D33, D33, D33, D33},
		  {D34, D34, D34, D34, D34, D34, D34, D34},
		  {D35, D35, D35, D35, D35, D35, D35, D35} }
	};

	const __m256 signbit = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
	__m256 max = _mm256_setzero_ps();
	__m256 a = _mm256_load_ps(x);
	for (unsigned i = 8; i <= len - (FILTER_LEN-1); i += 8) {
		 /* The input samples are shifted to produce the following sequence of
		  * registers.  The first column has samples 0-8, the next 1-9, and so on, but
		  * not in order.  The coefficient table matches the sequence.
		  * A  [  0  1  2  3  4  5  6  7 ]
		  * j1 [  1  2  3  8  5  6  7 12 ]
		  * j2 [  2  3  8  9  6  7 12 13 ]
		  * j3 [  3  8  9 10  7 12 13 14 ]
		  * r0 [  4  5  6  7  8  9 10 11 ]
		  * r1 [  5  6  7  4  9 10 11  8 ]
		  * r2 [  6  7  4  5 10 11  8  9 ]
		  * r3 [  7  4  5  6 11  8  9 10 ]
		  * B  [  8  9 10 11 12 13 14 15 ]
		  */

		__m256 b = _mm256_load_ps(x + i);
		__m256 t[PHASES];

		for_each_phase(k)
			t[k] = _mm256_mul_ps(a, c[0][k]);
		const __m256i ai = _mm256_castps_si256(a);
		const __m256i bi = _mm256_castps_si256(b);
		const __m256 j1 = _mm256_castsi256_ps(_mm256_alignr_epi8(bi, ai, 4));
		const __m256 j2 = _mm256_castsi256_ps(_mm256_alignr_epi8(bi, ai, 8));
		const __m256 j3 = _mm256_castsi256_ps(_mm256_alignr_epi8(bi, ai, 12));
		const __m256 r0 = _mm256_permute2f128_ps(a, b, 0x21);
		a = b;

		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(j1, c[1][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(j2, c[2][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(j3, c[3][k], t[k]);

		const __m256 r1 = _mm256_permute_ps(r0, 0x39);
		const __m256 r2 = _mm256_permute_ps(r0, 0x4e);
		const __m256 r3 = _mm256_permute_ps(r0, 0x93);

		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(r0, c[4][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(r1, c[5][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(r2, c[6][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(r3, c[7][k], t[k]);
		for_each_phase(k)
			t[k] = _mm256_fmadd_ps(a, c[8][k], t[k]);

		/* The 8 samples by 4 phases are now loaded in t[].  Each t[k] contains one
		 * phase of filter convoluted over the 15 samples, in order.  If we wanted to
		 * store the data in order we'd need to transpose it, but we don't need to do
		 * that to find the max. */

		// Absolute value
		for_each_phase(k)
			t[k] = _mm256_and_ps(t[k], signbit);
		// For each position find the max of the four phases
		__m256 pmax = _mm256_max_ps(_mm256_max_ps(t[0], t[1]), _mm256_max_ps(t[2], t[3]));
		// Find running max of all iterations
		max = _mm256_max_ps(pmax, max);
	}

	// Find horizontal max
	__m256 hmax = _mm256_max_ps(_mm256_permute2f128_ps(max, max, 0x01), max);
	hmax = _mm256_max_ps(hmax, _mm256_permute_ps(hmax, 0x4e));
	hmax = _mm256_max_ps(hmax, _mm256_permute_ps(hmax, 0xb1));
	return hmax[0];
}

#define findtppm	findtppm_avx
#define BLOCKSIZE	8

#else

static float findtppm_c(const float* x, unsigned len)
{
	static const float align(32) c[FILTER_LEN][PHASES] = {
	  {D00, D01, D02, D03 }, {D04, D05, D06, D07 }, {D08, D09, D10, D11 }, {D12, D13, D14, D15 },
	  {D16, D17, D18, D19 }, {D20, D21, D22, D23 }, {D24, D25, D26, D27 }, {D28, D29, D30, D31 },
	  {D32, D33, D34, D35 }
	};
	float max = 0.0f;

	for (unsigned i = 0; i < len - (FILTER_LEN-1); i++) {
		float y[PHASES] = {0.0f, };
		for (unsigned j = 0; j < FILTER_LEN; j++) {
			for_each_phase(k)
				y[k] += x[i+j] * c[j][k];
		}
		for_each_phase(k) {
			const float absy = fabs(y[k]);
			max = max < absy ? absy : max;
		}
	}
	return max;
}
#define findtppm	findtppm_c
#define BLOCKSIZE	1

#endif

struct tppm* tppm_init(unsigned chunk_size, unsigned max_age)
{
	struct tppm* tppm;
	tppm = calloc(1, sizeof(*tppm));
	if (!tppm)
		return NULL;
	const size_t bufsize = chunk_size * sizeof(*tppm->buffer);
#ifdef HAVE__ALIGNED_MALLOC
	tppm->buffer = _aligned_malloc(32, (bufsize + 31) & ~0x1f);
#else
	tppm->buffer = aligned_alloc(32, (bufsize + 31) & ~0x1f);
#endif

	if (!tppm->buffer) {
		free(tppm);
		return NULL;
	}
	tppm->deque = malloc(max_age * sizeof(*tppm->deque));
	if (!tppm->deque) {
		free(tppm->buffer);
		free(tppm);
		return NULL;
	}
	tppm->deque[0].value = 0;
	tppm->deque[0].time = 1;

	tppm->buffer_size = chunk_size;
	tppm->max_age = max_age;
	return tppm;
}

void tppm_free(struct tppm* tppm)
{
#ifdef HAVE__ALIGNED_MALLOC
	_aligned_free(tppm->buffer);
#else
	free(tppm->buffer);
#endif
	free(tppm->deque);
	free(tppm);
}

static void moving_window_max(struct tppm* tppm, float value)
{
	// Expire old max values from head
	while (tppm->tail != tppm->head && !time_after(tppm->deque[tppm->head].time, tppm->time))
		tppm->head = (tppm->head + tppm->max_age - 1) % tppm->max_age;
	assert(time_before(tppm->time, tppm->deque[tppm->head].time));

	// Drop values from tail that are <= new value
	tppm->tail = (tppm->tail - 1 + tppm->max_age) % tppm->max_age;
	while (tppm->tail != tppm->head) {
		unsigned next = (tppm->tail + 1) % tppm->max_age;
		if (value < tppm->deque[next].value)
			break;
		tppm->tail = next;
	}
	// Add new value at tail
	tppm->deque[tppm->tail].value = value;
	tppm->deque[tppm->tail].time = tppm->time + tppm->max_age;
	tppm->time++;
}

void tppm_process(struct tppm* tppm, const float* data, unsigned length)
{
	while (length > 0) {
		unsigned chunk = MIN(tppm->buffer_size - tppm->offset, length);
		memcpy(tppm->buffer + tppm->offset, data, chunk * sizeof(*data));
		data += chunk;
		length -= chunk;
		tppm->offset += chunk;

		if (tppm->offset == tppm->buffer_size) {
			const float peak = findtppm(tppm->buffer, tppm->buffer_size);
			moving_window_max(tppm, peak);
			tppm->peak = tppm->deque[tppm->head].value;
			// Age of peak is tppm->deque[tppm->head].time - tppm->time;

			// Copy unprocessed tail data to start of next chunk
			const unsigned tail = (tppm->buffer_size - (FILTER_LEN-1)) % BLOCKSIZE + (FILTER_LEN-1);
			memcpy(tppm->buffer, tppm->buffer + tppm->buffer_size - tail, tail * sizeof(*tppm->buffer));
			tppm->offset = tail;
		}
	}
}
