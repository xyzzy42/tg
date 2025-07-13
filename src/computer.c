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

static int count_events(const uint64_t *events, int wp, int nevents)
{
	int i;
	if(!nevents || !events[wp]) return 0;

	if(!events[0]) {
		for(i = 1; i < wp; i++)
			if(events[i]) break;
		return wp - i + 1;
	}
	for(i = wp+1; i < nevents; i++)
		if (events[i]) break;
	return nevents + wp - i + 1;
}

struct snapshot *snapshot_clone(struct snapshot *s)
{
	struct snapshot *t = malloc(sizeof(struct snapshot));
	memcpy(t,s,sizeof(struct snapshot));
	if(s->pb) t->pb = pb_clone(s->pb);
	t->events_count = count_events(s->events, s->events_wp, s->events_count);
	if(t->events_count) {
		t->events_wp = t->events_count - 1;
		t->events = malloc(t->events_count * sizeof(*t->events));
		t->events_tictoc = malloc(t->events_count * sizeof(*t->events_tictoc));
		int i, j;
		for(i = t->events_wp, j = s->events_wp; i >= 0; i--) {
			t->events[i] = s->events[j];
			t->events_tictoc[i] = s->events_tictoc[j];
			if(--j < 0) j = s->events_count - 1;
		}
	} else {
		t->events_wp = 0;
		t->events = NULL;
		t->events_tictoc = NULL;
	}
	t->amps_count = count_events(s->amps_time, s->amps_wp, s->amps_count);
	if(t->amps_count) {
		t->amps_wp = t->amps_count - 1;
		t->amps = malloc(t->amps_count * sizeof(*t->amps));
		t->amps_time = malloc(t->amps_count * sizeof(*t->amps_time));
		int i, j;
		for(i = t->amps_wp, j = s->amps_wp; i >= 0; i--) {
			t->amps[i] = s->amps[j];
			t->amps_time[i] = s->amps_time[j];
			if(--j < 0) j = s->amps_count - 1;
		}
	} else {
		t->amps_wp = 0;
		t->amps = NULL;
		t->amps_time = NULL;
	}
	if(s->d) {
		t->d = malloc(sizeof(*t->d));
		*t->d = *s->d;
	}
	return t;
}

void snapshot_destroy(struct snapshot *s)
{
	if(s->pb) pb_destroy_clone(s->pb);
	if(s->d) free(s->d);
	free(s->amps_time);
	free(s->amps);
	free(s->events_tictoc);
	free(s->events);
	free(s);
}

static int guess_bph(double period)
{
	double bph = 7200 / period;
	double min = bph;
	int i,ret;

	ret = 0;
	for(i=0; preset_bph[i]; i++) {
		double diff = fabs(bph - preset_bph[i]);
		if(diff < min) {
			min = diff;
			ret = i;
		}
	}

	return preset_bph[ret];
}

static void compute_update_cal(struct computer *c)
{
	c->actv->signal = analyze_processing_data_cal(c->pdata, c->cdata);
	if(c->actv->pb) {
		pb_destroy_clone(c->actv->pb);
		c->actv->pb = NULL;
	}
	c->actv->cal_state = c->cdata->state;
	c->actv->cal_percent = 100*c->cdata->wp/c->cdata->size;
	if(c->cdata->state == 1)
		c->actv->cal_result = round(10 * c->cdata->calibration);
}

static void compute_update(struct computer *c)
{
	struct processing_data *pd = c->pdata;
	struct processing_buffers *ps = pd->buffers;
	int step = pd->last_step;

	pd->last_step = 0;
	/* Do all buffers at once so that all computation interval(s) use the
	 * same data.  Buffers for some intervals will probably not be used, but
	 * it's not expensive to fill them.  Processing is the slow part.  */
	fill_buffers(ps);

	debug("\nSTART OF COMPUTATION CYCLE\n\n");
	unsigned int stepmask = BITMASK(NSTEPS); // Mask of available steps
	do {
		stepmask &= ~BIT(step);
		analyze_processing_data(c->pdata, step, c->actv->bph, c->actv->la, c->actv->events_from);

		if (ps[step].ready && ps[step].sigma < ps[step].period / 10000) {
			// Try next step if it's available
			if (stepmask & BIT(step+1)) step++;
		} else {
			// This step didn't pass, try a lesser step
			step--;
		}
	} while(step >= 0 && stepmask & BIT(step));

	if (step >= 0) {
		debug("%f +- %f\n", ps[step].period/ps[step].sample_rate, ps[step].sigma/ps[step].sample_rate);
		pd->last_tic = ps[step].last_tic;
		pd->last_step = step;

		if(c->actv->pb) pb_destroy_clone(c->actv->pb);
		c->actv->pb = pb_clone(&ps[step]);
		c->actv->is_old = 0;
		/* Signal's range is 0 to NSTEPS, while step is -1 to NSTEPS-1, i.e. signal = step+1 */
		c->actv->signal = step+1;
	} else {
		debug("---\n");
		c->actv->is_old = 1;
		c->actv->signal = 0;
	}
}

static void compute_events_cal(struct computer *c)
{
	struct calibration_data *d = c->cdata;
	struct snapshot *s = c->actv;
	int i;
	for(i=d->wp-1; i >= 0 &&
		d->events[i] > s->events[s->events_wp];
		i--);
	for(i++; i<d->wp; i++) {
		if(d->events[i] / s->nominal_sr <= s->events[s->events_wp] / s->nominal_sr)
			continue;
		if(++s->events_wp == s->events_count) s->events_wp = 0;
		s->events[s->events_wp] = d->events[i];
		debug("event at %llu\n",s->events[s->events_wp]);
	}
	s->events_from = get_timestamp();
}

static void compute_events(struct computer *c)
{
	struct snapshot *s = c->actv;
	struct processing_buffers *p = c->actv->pb;
	if(p && !s->is_old) {
		/* Add new events from p into s.  last is the timestamp where new events
		 * start.  It's a half-vibration after the last event, to avoid adding
		 * the same event twice with slightly different timestamps.  */
		const uint64_t last = s->events[s->events_wp] + floor(p->period / 4);
		bool newevent = false;
		int i;
		for(i=0; i<EVENTS_MAX && p->events[i].pos; i++)
			if(p->events[i].pos > last) {
				if(++s->events_wp == s->events_count) s->events_wp = 0;
				s->events[s->events_wp] = p->events[i].pos;
				s->events_tictoc[s->events_wp] = p->events[i].tictoc;
				debug("event at %llu\n", s->events[s->events_wp]);
				newevent = true;
			}

		/* Add a new amplitude if we have full signal, have new ticks, and it's been
		 * about a period since the last sample.  Place the amplitude measurement at the
		 * center of the averaging interval.  */
		if (newevent && s->signal == NSTEPS && p->amp > 0) {
			const uint64_t amp_time = p->timestamp - p->interval_count/2;
			if (amp_time > s->amps_time[s->amps_wp] + (int)(3*p->period/4)) {
				if(++s->amps_wp == s->amps_count) s->amps_wp = 0;
				s->amps[s->amps_wp] = p->amp;
				s->amps_time[s->amps_wp] = amp_time;
			}
		}

		s->events_from = p->timestamp - ceil(p->period);
	} else {
		s->events_from = get_timestamp();
	}
}

void compute_results(struct snapshot *s)
{
	s->sample_rate = s->nominal_sr * (1 + (double) s->cal / (10 * 3600 * 24));
	if(s->pb) {
		s->guessed_bph = s->bph ? s->bph : guess_bph(s->pb->period / s->sample_rate);
		s->rate = (7200/(s->guessed_bph * s->pb->period / s->sample_rate) - 1)*24*3600;
		s->be = fabs(s->pb->be) * 1000 / s->sample_rate;
		s->amp = s->la * s->pb->amp; // 0 = not available
		if(s->amp < 135 || s->amp > 360)
			s->amp = 0;
	} else
		s->guessed_bph = s->bph ? s->bph : DEFAULT_BPH;

	/* Find time, in beats, from last event to "now" */
	s->event_age = 0;
	if(s->events_count) {
		const uint64_t time = s->timestamp ? s->timestamp : get_timestamp();
		const uint64_t event = s->events[(s->events_wp + 1) % s->events_count];
		if(event) {
			const double beat_length = s->calibrate ? s->nominal_sr : (s->sample_rate * 3600) / s->guessed_bph;
			s->event_age = round((time - event) / beat_length);
		}
	}
}

static void *computing_thread(void *void_computer)
{
	struct computer *c = void_computer;
	for(;;) {
		pthread_mutex_lock(&c->mutex);
			while(!c->recompute)
				pthread_cond_wait(&c->cond, &c->mutex);
			if(c->recompute > 0) c->recompute = 0;
			int calibrate = c->calibrate;
			c->actv->bph = c->bph;
			c->actv->la = c->la;
			void (*callback)(void *) = c->callback;
			void *callback_data = c->callback_data;
		pthread_mutex_unlock(&c->mutex);

		if(c->recompute < 0) {
			if(callback) callback(callback_data);
			break;
		}

		if(calibrate && !c->actv->calibrate) {
			c->cdata->wp = 0;
			c->cdata->state = 0;
			c->actv->cal_state = 0;
			c->actv->cal_percent = 0;
		}
		if(calibrate != c->actv->calibrate)
			memset(c->actv->events,0,c->actv->events_count*sizeof(*c->actv->events));
		c->actv->calibrate = calibrate;

		if(c->actv->calibrate) {
			compute_update_cal(c);
			compute_events_cal(c);
		} else {
			compute_update(c);
			compute_events(c);
		}

		pthread_mutex_lock(&c->mutex);
			if(c->curr)
				snapshot_destroy(c->curr);
			if(c->clear_trace) {
				if(!calibrate) {
					memset(c->actv->events,0,c->actv->events_count*sizeof(*c->actv->events));
					memset(c->actv->amps,0,c->actv->amps_count*sizeof(*c->actv->amps));
					memset(c->actv->amps_time,0,c->actv->amps_count*sizeof(*c->actv->amps_time));
				}
				c->clear_trace = 0;
			}
			c->curr = snapshot_clone(c->actv);
		pthread_mutex_unlock(&c->mutex);

		if(callback) callback(callback_data);
	}

	debug("Terminating computation thread\n");

	return NULL;
}

void computer_destroy(struct computer *c)
{
	int i;
	for(i=0; i<NSTEPS; i++)
		pb_destroy(&c->pdata->buffers[i]);
	free(c->pdata->buffers);
	free(c->pdata);
	cal_data_destroy(c->cdata);
	free(c->cdata);
	snapshot_destroy(c->actv);
	if(c->curr)
		snapshot_destroy(c->curr);
	pthread_mutex_destroy(&c->mutex);
	pthread_cond_destroy(&c->cond);
	pthread_join(c->thread, NULL);
	free(c);
}

struct computer *start_computer(int nominal_sr, int bph, double la, int cal, int light)
{
	if(light) nominal_sr /= 2;
	set_audio_light(light);

	struct processing_buffers *p = malloc(NSTEPS * sizeof(*p));
	int first_step = light ? FIRST_STEP_LIGHT : FIRST_STEP;
	int i;
	for(i=0; i<NSTEPS; i++) {
		p[i].sample_rate = nominal_sr;
		p[i].interval_count = p[i].sample_count = nominal_sr * (1<<(i+first_step));
		setup_buffers(&p[i]);
	}

	struct processing_data *pd = malloc(sizeof(*pd));
	pd->buffers = p;
	pd->last_tic = 0;
	pd->is_light = light;
	pd->last_step = 0;

	struct calibration_data *cd = malloc(sizeof(*cd));
	setup_cal_data(cd);

	struct snapshot *s = malloc(sizeof(*s));
	s->timestamp = 0;
	s->nominal_sr = nominal_sr;
	s->pb = NULL;
	s->is_old = 1;
	s->calibrate = 0;
	s->signal = 0;
	s->events_count = EVENTS_COUNT;
	s->events = calloc(EVENTS_COUNT, sizeof(*s->events));
	s->events_tictoc = calloc(EVENTS_COUNT, sizeof(*s->events_tictoc));
	s->events_wp = 0;
	s->events_from = 0;
	s->amps_count = EVENTS_COUNT/2;
	s->amps = calloc(EVENTS_COUNT/2, sizeof(*s->amps));
	s->amps_time = calloc(EVENTS_COUNT/2, sizeof(*s->amps_time));
	s->amps_wp = 0;
	s->bph = bph;
	s->la = la;
	s->cal = cal;
	s->is_light = light;
	s->d = NULL;

	struct computer *c = malloc(sizeof(*c));
	c->cdata = cd;
	c->pdata = pd;
	c->actv = s;
	c->curr = snapshot_clone(s);
	c->recompute = 0;
	c->calibrate = 0;
	c->clear_trace = 0;

	if(    pthread_mutex_init(&c->mutex, NULL)
	    || pthread_cond_init(&c->cond, NULL)
	    || pthread_create(&c->thread, NULL, computing_thread, c)) {
		error("Unable to initialize computing thread");
		return NULL;
	}

	return c;
}

void lock_computer(struct computer *c)
{
	pthread_mutex_lock(&c->mutex);
}

void unlock_computer(struct computer *c)
{
	if(c->recompute)
		pthread_cond_signal(&c->cond);
	pthread_mutex_unlock(&c->mutex);
}
