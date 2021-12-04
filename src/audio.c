/*
    tg
    Copyright (C) 2015 Marcello Mamino

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

#include "tg.h"
#include <portaudio.h>
#include <errno.h>

/* Huge buffer of audio */
static float *pa_buffers;
static unsigned int pa_buffer_size;
static unsigned int write_pointer = 0;
static uint64_t timestamp = 0;
static pthread_mutex_t audio_mutex;


/** Filter chain.
 *
 */
struct filter_chain {
	struct biquad_filter *filters;
	unsigned int count;
	unsigned int max;
	pthread_mutex_t mutex;
	double sample_rate;
};

/** Data for PA callback to use */
struct callback_info {
	int 	channels;	//!< Number of channels
	bool	light;		//!< Light algorithm in use, copy half data
	struct filter_chain *chain; //!< Audio filter chain, e.g. hpf
};

/** Static object for audio device state.
 * There are calls that need this from the audio callback thread, the GUI thread, and
 * the computer thread.  Having each thread pass it in correctly would be really hard.
 * It's better to maintain it in one place here in the audio code.  Lacking class scope
 * in C, we'll have to settle for static global scope.  We only support once device at a
 * time, so not supporting multiuple audio contexts isn't much of a drawback.
 * */
static struct audio_context {
	PaStream *stream;	//!< Audio input stream object
	int device;  		//!< PortAudio device ID number
	int sample_rate;	//!< Requested sample rate (actual may differ)
	double real_sample_rate;//!< Real rate as returned by PA
	unsigned num_devices;	//!< Number of audio devices for current driver
	struct audio_device *devices;  //!< Cached audio device info
	//! Data callback will read, need to take care when modifying so as not to race.
	struct callback_info info;
} actx = {
	.device = -1,
};

/** Return effective sample rate.
 * This takes into account the half speed decimation enabled by light mode */
static inline double effective_sr(void)
{
	return actx.real_sample_rate / (actx.info.light ? 2 : 1);
}

/* Apply a biquadratic filter to data.  The delay values are updated in f, so
 * that it is possible to process data in chunks using multiple calls.
 */
static void apply_biquad(struct biquad_filter *f, float *data, unsigned int count)
{
	unsigned int i;
	double z1 = f->z1, z2 = f->z2;
	for(i=0; i<count; i++) {
		double in = data[i];
		double out = in * f->f.a0 + z1;
		z1 = in * f->f.a1 + z2 - f->f.b1 * out;
		z2 = in * f->f.a2 - f->f.b2 * out;
		data[i] = out;
	}
	f->z1 = z1;
	f->z2 = z2;
}

/* Functions for working on a chain of biquad filters.  A chain contains zero or more filters. 
 * It will grow as needed.  There are functions to insert, remove, access, enable, and apply
 * filters.  A filter may be in the chain but not be enabled, which means it will be skipped
 * when the chain is applied to audio.
 *
 * The chain is designed to be used simultaneously from two threads and internally contains the
 * necessary locking.  The manner in which the two threads access the chain is different:  One
 * thread is an audio processing thread the other a control thread. 
 *
 * The audio processing thread:
 *   Applies the filters to audio, but not does not change the chain itself.
 *   Has read/write access to the delay taps, but read-only access to all other fields.
 *   Does not care about any values of unenabled filters.
 *   Will acquire the chain's lock for its entire access to the filter.
 *   May hold the lock for a longer time.
 *
 * The control thread:
 *   Has read/write access to the eveything in the chain, but should not access the delay taps.
 *   Does not acquire the chain's lock for read-only accesses to the chain.
 *   Can perform read/write access on an unenabled filter without a lock (but can not delete the
 *   filter).
 *   Only acquires the lock for very short periods of time.
 *
 * This design is meant to allow a read-time audio processing thread the least chance of being
 * blocked by access from the control thread.  The control thread needs to lock to query the
 * chain and when modifying it needs a brief lock to insert new blank filter or disable an
 * existing filter, which can then be modified without a lock while disabled, and the a brief
 * lock to enable the filter.  */

/* Create a new blank chain */
struct filter_chain *filter_chain_init(double sample_rate)
{
	struct filter_chain *chain = malloc(sizeof(*chain));
	chain->filters = malloc(sizeof(*chain->filters));
	chain->count = 0;
	chain->max = 1;
	chain->sample_rate = sample_rate;
	pthread_mutex_init(&chain->mutex, NULL);
	return chain;
}

/* Insert empty non-enabled filter at index.  The index should be at most 1 greater than the
 * index of the last filter, i.e. no gaps.  (unsigned)-1 will insert at the end.  Returns a
 * pointer to the new filters (or NULL on failure, which should not happena.) */
struct biquad_filter *filter_chain_insert(struct filter_chain *chain, unsigned index)
{
	if (index == (unsigned)-1) {
		index = chain->count;
	} else if (index > chain->count) {
		printf("Insert after end of chain! Index %u with only %u filters\n", index, chain->count);
		return NULL;
	}

	pthread_mutex_lock(&chain->mutex);
	if (chain->count == chain->max) {
		chain->max = (unsigned)(chain->max * 1.5) + 1;
		/* Doing this re-alloc while holding the lock is a flaw.  We should allocate the
		 * memory with the lock released, then copy with it held.  We could even copy
		 * everything but the delay taps with it not held.  */
		chain->filters = realloc(chain->filters, chain->max * sizeof(*chain->filters));
	}
	memcpy(&chain->filters[index+1], &chain->filters[index], sizeof(*chain->filters) * (chain->count - index));
	chain->filters[index].enabled = false;
	chain->count++;
	pthread_mutex_unlock(&chain->mutex);

	memset(&chain->filters[index], 0, sizeof(chain->filters[index]));

	return &chain->filters[index];
}

/* Remove a filter from the chain.  It needs to exist in the chain. */
void filter_chain_remove(struct filter_chain *chain, unsigned index)
{
	if (index >= chain->count)
		return;
	pthread_mutex_lock(&chain->mutex);
	memcpy(&chain->filters[index], &chain->filters[index+1], sizeof(*chain->filters) * (chain->count - index - 1));
	chain->count--;
	pthread_mutex_unlock(&chain->mutex);
}

/* Return pointer to filter.  Returns NULL if i is past the end of available filters */
const struct biquad_filter *filter_chain_get(const struct filter_chain *chain, unsigned index)
{
	if (index >= chain->count)
		return NULL;
	return &chain->filters[index];
}

unsigned int filter_chain_count(const struct filter_chain *chain)
{
	return chain->count;
}

/* Enable or disable a filter */
bool filter_chain_enable(struct filter_chain *chain, unsigned index, bool enable)
{
	if (index >= chain->count || chain->filters[index].enabled == enable)
		return false;
	pthread_mutex_lock(&chain->mutex);
	chain->filters[index].enabled = enable;
	pthread_mutex_unlock(&chain->mutex);
	return true;
}

/* Program a filter */
static void _filter_set_coefficients(struct filter *f, enum bitype type, double f0_fS, double q, double gain)
{
	switch (type) {
		case HIGHPASS: make_hpq(f, f0_fS, q); break;
		case LOWPASS: make_lpq(f, f0_fS, q); break;
		case BANDPASS: make_bp(f, f0_fS, q); break;
		case NOTCH: make_notch(f, f0_fS, q); break;
		case ALLPASS: make_ap(f, f0_fS, q); break;
		case PEAK: make_peak(f, f0_fS, q, gain); break;
		default: break;
	}
}

/* Set or update a filter.  Takes the chain lock for enabled filters, but locking isn't
 * needed for disabled filters.  Returns false if the filter didn't need to be updated.  */
bool filter_chain_set_filter(struct filter_chain *chain, struct biquad_filter *filter,
	enum bitype type, unsigned freq, double q, double gain)
{
	if (filter->type == type && filter->frequency == freq && filter->bw == q && filter->gain == gain)
		return false;

	const double f0_fS = freq / chain->sample_rate;

	if (filter->enabled)
		pthread_mutex_lock(&chain->mutex);

	filter->type = type;
	filter->frequency = freq;
	filter->bw = q;
	filter->gain = gain;
	_filter_set_coefficients(&filter->f, type, f0_fS, q, gain);

	if (filter->enabled)
		pthread_mutex_unlock(&chain->mutex);

	return true;
}

/* Like filter_chain_set_filter(), but using index rather than a filter pointer */
bool filter_chain_set(struct filter_chain *chain, unsigned index,
	enum bitype type, unsigned freq, double q, double gain)
{
	if (index >= chain->count)
		return false;
	struct biquad_filter *filter = &chain->filters[index];
	return filter_chain_set_filter(chain, filter, type, freq, q, gain);
}

/* Move filter in position src to position dst */
void filter_chain_move(struct filter_chain *chain, unsigned src, unsigned dst)
{
	if (src == dst || src >= chain->count || dst >= chain->count)
		return;
	struct biquad_filter *newf = filter_chain_insert(chain, dst);
	if (dst <= src)
		src++;

	pthread_mutex_lock(&chain->mutex);
	memcpy(newf, &chain->filters[src], sizeof(*newf));
	chain->filters[src].enabled = false;
	pthread_mutex_unlock(&chain->mutex);
	filter_chain_remove(chain, src);
}

/* Adjusts existing filters to new sample rate.  Does nothing if rate is unchanged. */
static bool filter_chain_rate(struct filter_chain *chain, double sample_rate)
{
	if (chain->sample_rate == sample_rate)
		return false;

	/* We are lazy with locking, only need it around enabled filters */
	pthread_mutex_lock(&chain->mutex);
	unsigned i;
	for (i = 0; i < chain->count; i++) {
		struct biquad_filter *filter = &chain->filters[i];
		_filter_set_coefficients(&filter->f, filter->type, filter->frequency / sample_rate, filter->bw, filter->gain);
	}
	pthread_mutex_unlock(&chain->mutex);

	chain->sample_rate = sample_rate;
	return true;
}

/* Apply filter chain to audio data.  Data is fixed as the global array pa_buffers.  Process
 * data over range [start, stop).  This might wrap.  */
static void filter_chain_apply(struct filter_chain *chain, float *data, unsigned start, unsigned stop, unsigned size)
{
	unsigned i;

	/* Apply each filter in turn on a single contiguous chunk of the ciruclar buffer,
	   of which there is either 1 or 2 chunks. */
	const unsigned len = start < stop ? stop - start : size - start;

	pthread_mutex_lock(&chain->mutex);
	// First chunk is either [start, stop) or [start, size)
	for (i = 0; i < chain->count; i++)
		if (chain->filters[i].enabled)
			apply_biquad(&chain->filters[i], data + start, len);
	// Second chunk, when wrapped, is [0, stop)
	if (stop <= start)
		for (i = 0; i < chain->count; i++)
			if (chain->filters[i].enabled)
				apply_biquad(&chain->filters[i], data, stop);
	pthread_mutex_unlock(&chain->mutex);
}

static int paudio_callback(const void *input_buffer,
			   void *output_buffer,
			   unsigned long frame_count,
			   const PaStreamCallbackTimeInfo *time_info,
			   PaStreamCallbackFlags status_flags,
			   void *data)
{
	UNUSED(output_buffer);
	UNUSED(time_info);
	UNUSED(status_flags);
	const float *input_samples = (const float*)input_buffer;
	unsigned long i;
	const struct callback_info *info = data;
	unsigned wp = write_pointer;

	if (info->light) {
		static bool even = true;
		/* Copy every other sample.  It would be much more efficient to
		 * just drop the sample rate if the sound hardware supports it.
		 * This would also avoid the aliasing effects that this simple
		 * decimation without a low-pass filter causes.  */
		if(info->channels == 1) {
			for(i = even ? 0 : 1; i < frame_count; i += 2) {
				pa_buffers[wp++] = input_samples[i];
				if (wp >= pa_buffer_size) wp -= pa_buffer_size;
			}
		} else {
			for(i = even ? 0 : 2; i < frame_count*2; i += 4) {
				pa_buffers[wp++] = input_samples[i] + input_samples[i+1];
				if (wp >= pa_buffer_size) wp -= pa_buffer_size;
			}
		}
		/* Keep track if we have processed an even number of frames, so
		 * we know if we should drop the 1st or 2nd frame next callback. */
		if(frame_count % 2) even = !even;
	} else {
		const unsigned len = MIN(frame_count, pa_buffer_size - wp);
		if(info->channels == 1) {
			memcpy(pa_buffers + wp, input_samples, len * sizeof(*pa_buffers));
			if(len < frame_count)
				memcpy(pa_buffers, input_samples + len, (frame_count - len) * sizeof(*pa_buffers));
		} else {
			for(i = 0; i < len; i++)
				pa_buffers[wp + i] = input_samples[2u*i] + input_samples[2u*i + 1u];
			if(len < frame_count)
				for(i = len; i < frame_count; i++)
					pa_buffers[i - len] = input_samples[2u*i] + input_samples[2u*i + 1u];
		}
		wp = (wp + frame_count) % pa_buffer_size;
	}

	filter_chain_apply(info->chain, pa_buffers, write_pointer, wp, pa_buffer_size);

	pthread_mutex_lock(&audio_mutex);
	write_pointer = wp;
	timestamp += frame_count;
	pthread_mutex_unlock(&audio_mutex);
	return 0;
}

static PaError open_stream(PaDeviceIndex index, unsigned int rate, bool light, PaStream **stream)
{
	PaError err;

	long channels = Pa_GetDeviceInfo(index)->maxInputChannels;
	if(channels == 0) {
		error("Default audio device has no input channels");
		return paInvalidChannelCount;
	}
	if(channels > 2) channels = 2;
	actx.info.channels = channels;
	actx.info.light = light;

	err = Pa_OpenStream(stream,
			    &(PaStreamParameters){
			            .device = index,
				    .channelCount = channels,
				    .sampleFormat = paFloat32,
				    .suggestedLatency = Pa_GetDeviceInfo(index)->defaultHighInputLatency,
			    },
		            NULL,
			    rate,
			    paFramesPerBufferUnspecified,
			    paNoFlag,
			    paudio_callback,
			    &actx.info);
	return err;
}

/** Select audio device and enable recording.
 *
 * This will select `device` to be the active audio device and capture at the
 * rate provided in `*nominal_sr`.  If `*normal_sr` is zero, then a default rate
 * is selected.
 *
 * It is safe to call if the device and rate are unchanged.  This will be
 * detected and nothing will be done.
 *
 * Light mode will use simple decimation to cut the sample rate in half.  The
 * values in nominal and real sr do not reflect this.
 *
 * @param device Device number, index of device from get_audio_devices() list
 * @param[in,out] normal_sr Desired rate, or zero for default.  Rate used on return.
 * @param[out] real_sr Actual exact rate received, might be different than nominal_sr.
 * @param chain The filter chain for the audio.  NULL continues to use the existing chain or creates an empty one.
 * When changing devices after the first open, only NULL is supported.
 * @param light Use light mode (halve normal_sr)
 * @returns zero or one on success or negative error code.  1 indicates no
 * change in device or rate was needed.
 */
int set_audio_device(int device, int *nominal_sr, double *real_sr, struct filter_chain *chain, bool light)
{
	PaError err;

	// FIXME: Use a list of rates and pick the first supported rate
	if(*nominal_sr == 0)
		*nominal_sr = PA_SAMPLE_RATE;

	if (actx.info.chain && chain) {
		// Don't need to support this
		printf("Don't support changing filter chain after audio already running\n");
		return -EBUSY;
	}

	if(actx.device == device && actx.sample_rate == *nominal_sr) {
		if(real_sr) *real_sr = actx.real_sample_rate;
		return 1; // Already using this device at this rate
	}

	if(actx.device != -1) {
		// Stop current device
		Pa_StopStream(actx.stream);
		Pa_CloseStream(actx.stream);
		actx.stream = NULL;
		actx.device = -1;
	}

	actx.sample_rate = *nominal_sr;

	// Start new one.  It seems it doesn't succeed on the first try sometimes.
	unsigned int n;
	for(n = 5; n; n--) {
		debug("Open device %d at %d Hz with %d tries left\n", device, actx.sample_rate, n);
		err = open_stream(device, actx.sample_rate, light, &actx.stream);
		if (err == paNoError)
			break;
		if (err != paDeviceUnavailable)
			goto error;
		usleep(500000);
	}
	if(!n)
		goto error;
	actx.real_sample_rate = Pa_GetStreamInfo(actx.stream)->sampleRate;

	// This assumes changing an existing chain isn't supported
	if (actx.info.chain == NULL) {
		// Install empty chain if none given
		actx.info.chain = chain ? chain : filter_chain_init(effective_sr());
	}
	// Keep using existing chain, adjust for sample rate change if needed
	filter_chain_rate(actx.info.chain, effective_sr());

	/* Allocate larger buffer if needed */
	const size_t buffer_size = actx.sample_rate << (NSTEPS + FIRST_STEP);
	if(pa_buffer_size < buffer_size) {
		if(pa_buffers) free(pa_buffers);
		pa_buffers = calloc(buffer_size, sizeof(*pa_buffers));
		if(!pa_buffers) {
			err = paInsufficientMemory;
			goto error;
		}
		pa_buffer_size = buffer_size;
	}

	err = Pa_StartStream(actx.stream);
	if(err != paNoError) {
		Pa_CloseStream(actx.stream);
		goto error;
	}

	/* Return sample rates used */
	*nominal_sr = actx.sample_rate;
	if(real_sr)
		*real_sr = actx.real_sample_rate;

	actx.device = device;
	return 0;

error:
	actx.stream = NULL;
	actx.device = -1;
	actx.sample_rate = 0;
	actx.real_sample_rate = 0.0;

	const struct PaDeviceInfo* devinfo = Pa_GetDeviceInfo(device);
	const char *err_str = Pa_GetErrorText(err);
	error("Error opening audio device '%s' at %d Hz: %s", devinfo->name, *nominal_sr, err_str);
	return err;
}

/** Return current audio device.
 *
 * @return Index of current audio device, or -1 if none is active.
 */
int get_audio_device(void)
{
	return actx.device;
}

/** Get list of devices.
 *
 * @param[out] devices Static list of devices.
 * @return Number of devices or negative on error.
 */
int get_audio_devices(const struct audio_device **devices)
{
	const struct audio_device* devs = actx.devices;
	*devices = devs;
	return actx.num_devices;
}

static bool check_audio_rate(int device, int rate)
{
	const long channels = Pa_GetDeviceInfo(device)->maxInputChannels;
	const PaStreamParameters params = {
		.device = device,
		.channelCount = channels > 2 ? 2 : channels,
		.sampleFormat = paFloat32,
	};

	return paFormatIsSupported == Pa_IsFormatSupported(&params, NULL, rate);
}

static void scan_audio_device(int i, bool dopulse)
{
	static const int rate_list[] = AUDIO_RATES;
	const struct PaDeviceInfo* devinfo = Pa_GetDeviceInfo(i);

	debug("Device %2d: %2d %s%s\n", i, devinfo->maxInputChannels, devinfo->name, i == Pa_GetDefaultInputDevice() ? " (default)" : "");
	bool ispulse = !strcmp(devinfo->name, "pulse") || !strcmp(devinfo->name, "default") || !strcmp(devinfo->name, "sysdefault");
	if (ispulse ^ dopulse)
		return;
	actx.devices[i].name = devinfo->name;
	actx.devices[i].good = devinfo->maxInputChannels > 0;
	actx.devices[i].isdefault = i == Pa_GetDefaultInputDevice();
	actx.devices[i].rates = 0;
	if (actx.devices[i].good) {
		unsigned r;
		for (r = 0; r < ARRAY_SIZE(rate_list); r++)
			if (check_audio_rate(i, rate_list[r]))
				actx.devices[i].rates |= 1 << r;
	}
}

static int scan_audio_devices(void)
{
	const int n = Pa_GetDeviceCount();

	if (actx.devices) free(actx.devices);
	actx.devices = calloc(n, sizeof(*actx.devices));
	if (!actx.devices)
		return -ENOMEM;

	// Two pass, non-pulse followed by pulse, as scanning a pulseaudio device will make the
	// real device it's using appear busy if it's scanned shortly afterward.
	int i;
	for (i = 0; i < n; i++)
		scan_audio_device(i, false);
	for (i = 0; i < n; i++)
		scan_audio_device(i, true);
	actx.num_devices = n;

	return n;
}

/** Start audio system.
 *
 * This will start the recording stream.  Call this first before any other audio
 * functions, as it initialize PortAudio and fills in the device list.
 *
 * If the selected device is not suitable, perhaps because the audio hardware has
 * changed since the device number was saved, it will fallback to the default device.
 *
 * A sample rate of 0 will select the default sample rate.
 *
 * On error, audio is NOT running.
 *
 * The distinction between the nominal and real sample rate is somewhat ill-defined.
 * Nothing uses real sample rate yet.
 *
 * @param device The device to use, or -1 for default.
 * @param[in,out] normal_sample_rate The rate in Hz to use, or 0 for default.  Returns
 * actual rate selected.
 * @param[out] real_sample_rate The exact rate used.
 * @param chain Filter chain to use.  NULL will create an empty chain.
 * @param light Use light mode (decimate to half supplied rate).
 * @returns 0 on success, 1 on error.
 *
 */
int start_portaudio(int device, int *nominal_sample_rate, double *real_sample_rate, struct filter_chain *chain, bool light)
{
	if(pthread_mutex_init(&audio_mutex,NULL)) {
		error("Failed to setup audio mutex");
		return 1;
	}

	PaError err = Pa_Initialize();
	if(err!=paNoError) {
		error("Error initializing PortAudio: %s", Pa_GetErrorText(err));
		goto error;
	}

#ifdef DEBUG
	if(testing) {
		*nominal_sample_rate = PA_SAMPLE_RATE;
		*real_sample_rate = PA_SAMPLE_RATE;
		goto end;
	}
#endif

	if(scan_audio_devices() < 0) {
		error("Unable to query audio devices");
		// Maybe default audio device will work anyway?
	}

	PaDeviceIndex input;
	// Use default input if no device selected or selected device is no longer available.
	if(device < 0 || device >= (int)actx.num_devices || !actx.devices[device].good) {
		input = Pa_GetDefaultInputDevice();
		if(input == paNoDevice) {
			error("No default audio input device found");
			goto error;
		}
	} else
		input = device;

	err = set_audio_device(input, nominal_sample_rate, real_sample_rate, chain, light);
	if(err!=paNoError && err!=1)
		goto error;

#ifdef DEBUG
end:
#endif
	debug("sample rate: nominal = %d real = %f\n",*nominal_sample_rate,*real_sample_rate);

	return 0;

error:
	error("Unable to start audio");
	return 1;
}

int terminate_portaudio()
{
	debug("Closing portaudio\n");
	PaError err = Pa_Terminate();
	if(err != paNoError) {
		error("Error closing audio: %s", Pa_GetErrorText(err));
		return 1;
	}
	return 0;
}

uint64_t get_timestamp()
{
	pthread_mutex_lock(&audio_mutex);
	uint64_t ts = actx.info.light ? timestamp / 2 : timestamp;
	pthread_mutex_unlock(&audio_mutex);
	return ts;
}

void fill_buffers(struct processing_buffers *ps)
{
	pthread_mutex_lock(&audio_mutex);
	uint64_t ts = timestamp / (actx.info.light ? 2 : 1);
	int wp = write_pointer;
	pthread_mutex_unlock(&audio_mutex);

	int i;
	for(i = 0; i < NSTEPS; i++) {
		ps[i].timestamp = ts;

		int start = wp - ps[i].sample_count;
		if (start < 0) start += pa_buffer_size;
		int len = MIN((unsigned)ps[i].sample_count, pa_buffer_size - start);
		memcpy(ps[i].samples, pa_buffers + start, len * sizeof(*pa_buffers));
		if (len < ps[i].sample_count)
			memcpy(ps[i].samples + len, pa_buffers, (ps[i].sample_count - len) * sizeof(*pa_buffers));
	}
}

/** Change to light mode
 *
 * Call to enable or disable light mode.  Changing the mode will empty the audio
 * buffer.  Nothing will happen if the mode doesn't actually change.  Audio is
 * downsampled by 2 in light mode.
 *
 * @param light True for light mode, false for normal mode
 */
void set_audio_light(bool light)
{
	if(actx.info.light != light) {
		Pa_StopStream(actx.stream);
		pthread_mutex_lock(&audio_mutex);

		actx.info.light = light;
		memset(pa_buffers, 0, sizeof(*pa_buffers) * pa_buffer_size);
		write_pointer = 0;
		timestamp = 0;

		pthread_mutex_unlock(&audio_mutex);

		filter_chain_rate(actx.info.chain, effective_sr());
		PaError err = Pa_StartStream(actx.stream);
		if(err != paNoError)
			error("Error re-starting audio input: %s", Pa_GetErrorText(err));
	}
}


static float* _get_audio_data(uint64_t *start_time, unsigned int len)
{
	float *data = malloc(sizeof(*data) * len);
	uint64_t time = *start_time;

	/* This copies the data with the mutex held, unlike fill_buffers(), which gets
	 * the timestamp while holding the mutex but does the coping unlocked. 
	 * fill_buffers() only uses at most the 1st half of the audio ring buffer, so
	 * the audio thread will need to write over half the ring buffer to modify the
	 * audio being copied, i.e. it has 16 seconds to make the copy.  This function
	 * could get audio from the end of the ring buffer, which might be modified on
	 * the next audio thread callback.  */
	pthread_mutex_lock(&audio_mutex);
	const uint64_t ts = actx.info.light ? timestamp / 2 : timestamp;
	const uint64_t audio_end = ts > pa_buffer_size ? ts - pa_buffer_size : 0;

	// time -1 means from the end
	if (time == (uint64_t)-1)
		*start_time = time = ts - len;

	if (time + len > ts || time < audio_end) {
		pthread_mutex_unlock(&audio_mutex);
		free(data);
		return NULL;
	}

	const int start = (write_pointer + pa_buffer_size - (ts - time)) % pa_buffer_size;
	const unsigned chunk = MIN(len, pa_buffer_size - start);
	memcpy(data, pa_buffers + start, chunk * sizeof(*pa_buffers));
	if (chunk < len)
		memcpy(data + chunk, pa_buffers, (len - chunk) * sizeof(*pa_buffers));

	pthread_mutex_unlock(&audio_mutex);
	return data;
}

float* get_audio_data(uint64_t start_time, unsigned int len)
{
	return _get_audio_data(&start_time, len);
}

float* get_last_audio_data(unsigned int len, uint64_t* timestamp)
{
	*timestamp = (uint64_t)-1;
	return _get_audio_data(timestamp, len);
}
