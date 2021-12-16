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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <fftw3.h>
#include <stdarg.h>
#include <gtk/gtk.h>
#include <pthread.h>

#ifdef __CYGWIN__
#define _WIN32
#endif

#define CONFIG_FILE_NAME "tg-timer.ini"

#define FILTER_CUTOFF 3000

#define CAL_DATA_SIZE 900

#define FIRST_STEP 1
#define FIRST_STEP_LIGHT 0

#define NSTEPS 4
#define PA_SAMPLE_RATE 44100u

#define OUTPUT_FONT 40
#define OUTPUT_WINDOW_HEIGHT 70

#define POSITIVE_SPAN 10
#define NEGATIVE_SPAN 25

#define EVENTS_COUNT 10000
#define EVENTS_MAX 100
#define PAPERSTRIP_ZOOM 10
#define PAPERSTRIP_ZOOM_CAL 100
#define PAPERSTRIP_MARGIN .2

#define MIN_BPH 8100
#define TYP_BPH 12000
#define MAX_BPH 72000
#define DEFAULT_BPH 21600
#define MIN_LA 10 // deg
#define MAX_LA 90 // deg
#define DEFAULT_LA 52 // deg
#define MIN_CAL -1000 // 0.1 s/d
#define MAX_CAL 1000 // 0.1 s/d

#define PRESET_BPH { 12000, 14400, 17280, 18000, 19800, 21600, 25200, 28800, 36000, 43200, 72000, 0 };

#ifdef DEBUG
#define debug(...) print_debug(__VA_ARGS__)
#else
#define debug(...) {}
#endif

#define UNUSED(X) (void)(X)
#define BIT(n) (1u << (n))
#define BITMASK(n) ((1u << (n)) - 1u)
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define time_after(a, b) ((int)((b) - (a)) < 0)
#define time_before(a, b) time_after(b, a)

/** Hold timestamp and type of a vibration. */
struct event {
	uint64_t pos;	//< Position, may be relative or absolute timestamp
	bool tictoc;	//< Is this vibration the tic or toc half-cycle?
};

/* algo.c */
struct processing_buffers {
	int sample_rate;	//< Nominal sampling rate in Hz
	int sample_count;	//< Number of samples in this buffer
	float *samples, *samples_sc, *waveform, *waveform_sc, *tic_wf, *slice_wf, *tic_c;
	fftwf_complex *fft, *sc_fft, *tic_fft, *slice_fft;
	fftwf_plan plan_a, plan_b, plan_c, plan_d, plan_e, plan_f, plan_g;
	struct filter *lpf;
	/** Number of samples following values were estimated from.
	 * This might by the same as sample_count, but can be less if the buffer was cloned and
	 * only one period of data was saved.  */
	int interval_count;	
	double period;		//< Estimated period (tic + toc) in samples
	double sigma,be,waveform_max,phase,tic_pulse,toc_pulse,amp;
	double cal_phase;
	int waveform_max_i;
	int tic,toc;
	int ready;		//< Boolean flag indicating buffer is good data
	uint64_t timestamp;	//< Absolute timestamp of end of buffer, timestamp - sample_count is time at start of buffer.
	uint64_t last_tic, last_toc, events_from;
	/** Dynamically allocated array of events, using absolute timestamps.  Terminated by 0 value for position. */
	struct event *events;
#ifdef DEBUG
	int debug_size;
	float *debug;
#endif
};

struct processing_data {
	struct processing_buffers *buffers;
	uint64_t last_tic;
	int last_step;	//!< Guess of step (buffers index) to try first, based on last iteration
	int is_light;
};

struct calibration_data {
	int wp;
	int size;
	int state;
	double calibration;
	uint64_t start_time;
	double *times;
	double *phases;
	uint64_t *events;
};

struct filter {
	double a0,a1,a2,b1,b2;
};

/** Filter types. */
enum bitype {
	LOWPASS,
	HIGHPASS,
	BANDPASS,
	NOTCH,
	ALLPASS,
	PEAK,
	CUSTOM,
	NUM_BITYPES
};

/** A biquadratic filter.
 * 
 * Saves the delay taps to allow the filter to continue across multiple calls. */
struct biquad_filter {
	struct filter f;	//!< Filter coefficients, F(z) = a(z) / b(z)
	double        z1, z2;	//!< Delay taps
	double	      bw;	//!< Bandwidth or Q
	double        gain;	//!< Gain (only for peaking filter)
	unsigned      frequency;//!< Cut-off or center frequency
	enum bitype   type;	//!< Filter type
	bool	      enabled;  //!< Enable filter
};

/** Opaque type for filter chain */
struct filter_chain;

void setup_buffers(struct processing_buffers *b);
void pb_destroy(struct processing_buffers *b);
struct processing_buffers *pb_clone(struct processing_buffers *p);
void pb_destroy_clone(struct processing_buffers *p);
void setup_cal_data(struct calibration_data *cd);
void cal_data_destroy(struct calibration_data *cd);
int test_cal(struct processing_buffers *p);
void make_hp(struct filter *f, double freq); // q = 1/√2
void make_hpq(struct filter *f, double freq, double q);
void make_lp(struct filter *f, double freq); // q = 1/√2
void make_lpq(struct filter *f, double freq, double q);
void make_bp(struct filter *f, double freq, double bw);
void make_ap(struct filter *f, double freq, double bw);
void make_notch(struct filter *f, double freq, double bw);
void make_peak(struct filter *f, double freq, double bw, double gain);
bool analyze_processing_data(struct processing_data *pd, int step, int bph, double la, uint64_t events_from);
int analyze_processing_data_cal(struct processing_data *pd, struct calibration_data *cd);

/* audio.c */
#define AUDIO_RATES       {22050,       44100,      48000,    96000,     192000 }
#define AUDIO_RATE_LABELS {"22.05 kHz", "44.1 kHz", "48 kHz", "96 kHz", "192 kHz" }
#define NUM_AUDIO_RATES ARRAY_SIZE((int[])AUDIO_RATES)

int start_portaudio(int device, int *nominal_sample_rate, double *real_sample_rate, struct filter_chain *chain, bool light);
int terminate_portaudio();
uint64_t get_timestamp();
void fill_buffers(struct processing_buffers *ps);
void set_audio_light(bool light);
struct audio_device {
	const char* name;      //!< Name of device from port audio
	bool        good;      //!< Is this suitable or not?  E.g., playback only.
	bool	    isdefault; //!< This is the default device;
	unsigned    rates;     //!< Bitmask of allowed rates from AUDIO_RATES

};
int get_audio_devices(const struct audio_device **devices);
int get_audio_device(void);
int set_audio_device(int device, int *nominal_sr, double *real_sr, struct filter_chain *chain, bool light);
struct filter_chain* get_audio_filter_chain(void);
float* get_audio_data(uint64_t start_time, unsigned int len);
float* get_last_audio_data(unsigned int len, uint64_t *timestamp);

float get_audio_peak(void);
void set_audio_tppm(bool enable);

struct filter_chain *filter_chain_init(double sample_rate);
struct biquad_filter *filter_chain_insert(struct filter_chain *chain, unsigned index);
void filter_chain_remove(struct filter_chain *chain, unsigned index);
const struct biquad_filter *filter_chain_get(const struct filter_chain *chain, unsigned index);
unsigned int filter_chain_count(const struct filter_chain *chain);
void filter_chain_move(struct filter_chain *chain, unsigned src, unsigned dst);
bool filter_chain_enable(struct filter_chain *chain, unsigned index, bool enable);
bool filter_chain_set(struct filter_chain *chain, unsigned index, enum bitype type, unsigned freq, double q, double gain);
bool filter_chain_set_filter(struct filter_chain *chain, struct biquad_filter *filter, enum bitype type, unsigned freq, double q, double gain);

/* computer.c */
struct display;

struct snapshot {
	struct processing_buffers *pb;
	int is_old;
	uint64_t timestamp;
	int is_light;

	int nominal_sr; // W/O calibration, but does include light mode decimation
	int calibrate;
	int bph;
	double la; // deg
	int cal; // 0.1 s/d

	int events_count;
	uint64_t *events; // used in cal+timegrapher mode
	unsigned char *events_tictoc;	//< Tic or Toc for each event
	int events_wp; // used in cal+timegrapher mode
	uint64_t events_from; // used only in timegrapher mode

	int signal;

	int cal_state;
	int cal_percent;
	int cal_result; // 0.1 s/d

	// data dependent on bph, la, cal
	double sample_rate; // Includes calibration
	int guessed_bph;
	double rate;
	double be;
	double amp;

	// State related to displaying the snapshot, not generated by computer
	struct display *d;
};

struct computer {
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t cond;

// controlled by interface
	int recompute;
	int calibrate;
	int bph;
	double la; // deg
	int clear_trace;
	void (*callback)(void *);
	void *callback_data;

	struct processing_data *pdata;
	struct calibration_data *cdata;

	struct snapshot *actv;
	struct snapshot *curr;
};

struct snapshot *snapshot_clone(struct snapshot *s);
void snapshot_destroy(struct snapshot *s);
void computer_destroy(struct computer *c);
struct computer *start_computer(int nominal_sr, int bph, double la, int cal, int light);
void lock_computer(struct computer *c);
void unlock_computer(struct computer *c);
void compute_results(struct snapshot *s);

/* output_panel.c */
/* Snapshot display parameters, e.g. scale, centering. */
struct display {
	// Scaling factor for each beat.  1 means the chart is 1 beat wide, 0.5
	// means half a beat, etc.
	double beat_scale;
	/* Time of point used to anchor the paperstrip.  Each paperstrip point's position is
	 * relative to the previous point.  This point is the one with an absolute position that
	 * is kept the same, so that all the dots do not shift side to side as the scroll.  */
	uint64_t anchor_time;
	// Phase offset of point at anchor_time
	double anchor_offset;
};

struct output_panel {
	GtkWidget *panel;

	GtkWidget *output_drawing_area;
	GtkWidget *displays;
	GtkWidget *waveforms_box;
	GtkWidget *tic_drawing_area;
	GtkWidget *toc_drawing_area;
	GtkWidget *period_drawing_area;
	GtkWidget *paperstrip_box;
	GtkWidget *paperstrip_drawing_area;
	GtkWidget *clear_button;
	GtkWidget *left_button;
	GtkWidget *right_button;
	GtkWidget *zoom_button;
	GtkWidget *zoom_orig_button;
#ifdef DEBUG
	GtkWidget *debug_drawing_area;
#endif
	bool vertical_layout;

	struct computer *computer;
	struct snapshot *snst;
};

void initialize_palette();
struct output_panel *init_output_panel(struct computer *comp, struct snapshot *snst, int border, bool vertical_layout);
void set_panel_layout(struct output_panel *op, bool vertical);
void redraw_op(struct output_panel *op);
void op_set_snapshot(struct output_panel *op, struct snapshot *snst);
void op_set_border(struct output_panel *op, int i);
void op_destroy(struct output_panel *op);

/* interface.c */
struct main_window {
	GtkApplication *app;

	GtkWidget *window;
	GtkWidget *bph_combo_box;
	GtkWidget *la_spin_button;
	GtkWidget *cal_spin_button;
	GtkWidget *snapshot_button;
	GtkWidget *snapshot_name;
	GtkWidget *snapshot_name_entry;
	GtkWidget *cal_button;
	GtkWidget *notebook;
	GtkWidget *save_item;
	GtkWidget *save_all_item;
	GtkWidget *close_all_item;

	/* Audio Setup dialog */
	GtkWidget *audio_setup;
	GtkComboBox *device_list;
	GtkComboBox *rate_list;
	GtkRange *hpf_range;

	/* Signal dialog */
	GtkWidget *signal_dialog;
	GtkWidget *signal_graph;
	GtkWidget *spectime_spin;

	GtkWidget *filter_chain_dialog;

	struct output_panel *active_panel;

	struct computer *computer;
	struct snapshot *active_snapshot;
	int computer_timeout;

	int is_light;
	int zombie;
	int controls_active;
	int calibrate;
	int bph;
	double la; // deg
	int cal; // 0.1 s/d
	int nominal_sr;	// requested audio device rate
	int audio_device;// Selected device
	int audio_rate;  // Selected rate
	int hpf_freq;    // Low-pass filter cutoff frequency
	struct filter_chain *filter_chain; // To be passed to audio startup

	bool vertical_layout;

	GKeyFile *config_file;
	gchar *config_file_name;
	struct conf_data *conf_data;

	guint kick_timeout;
	guint save_timeout;
};

extern int preset_bph[];

#ifdef DEBUG
extern int testing;
#endif

void print_debug(char *format,...);
void error(char *format,...);
void recompute(struct main_window *w);

/* audio_interface.c */
void init_audio_dialog(struct main_window *w);
void audio_setup(GtkMenuItem *m, struct main_window *w);

/* config.c */
#define CONFIG_FIELDS(OP) \
	OP(bph, bph, int) \
	OP(lift_angle, la, double) \
	OP(calibration, cal, int) \
	OP(light_algorithm, is_light, int) \
	OP(vertical_paperstrip, vertical_layout, bool) \
	OP(audio_device, audio_device, int) \
	OP(audio_rate, audio_rate, int) \
	OP(highpass_cutoff_freq, hpf_freq, int)

struct conf_data {
#define DEF(NAME,PLACE,TYPE) TYPE PLACE;
	CONFIG_FIELDS(DEF)
};

void load_config(struct main_window *w);
void save_config(struct main_window *w);
void save_on_change(struct main_window *w);
void close_config(struct main_window *w);

/* serializer.c */
int write_file(FILE *f, struct snapshot **s, char **names, uint64_t cnt);
int read_file(FILE *f, struct snapshot ***s, char ***names, uint64_t *cnt);

/* python.c */
#ifdef HAVE_PYTHON
bool python_init(const struct main_window* w);
void python_finish(void);
#else
static inline bool python_init(const struct main_window* w) { UNUSED(w); return true; }
static inline void python_finish(void) { }
#endif
void create_filter_plot(GtkImage* image, const struct filter* filter, int f0, int Fs, double Q);
void create_filter_chain_plot(GtkImage *image);
void create_filter_n_plot(GtkImage *image, unsigned n);
void spectrogram_beat(struct main_window *w, int which);
void spectrogram_time(struct main_window *w, double time_sec);
void image_set_minimum_size(GtkImage* widget, int width, int height);

/* filter_interface.c */
GtkWidget* spin_slider_new(const char* label, gdouble min, gdouble max, gdouble step);
typedef struct _FilterDialog FilterDialog;
GtkWidget* filter_dialog_new(struct main_window *w);
void filter_dialog_set_chain(FilterDialog* filter_dialog, struct filter_chain* chain);

/* tppm.c */
/** True Peak Programme Meter.
 *
 * Attempt to find max audio level, for adjusting gain and pre-amplifiers.
 */
struct tppm {
	float peak;		//!< Current max value in moving window

	// For filtering and finding max of each chunk
	float *buffer;		//!< Buffer of data to be processing
	unsigned buffer_size;	//!< Size of buffer
	unsigned offset;	//!< Current buffer pointer

	// Moving window maximum of chunk maxes
	unsigned max_age;	//!< Time at which peak expires
	struct {
		float value;	//!< Value of element
		unsigned time;	//!< Expire time of element
	} *deque;		//!< Sorted list of local maxes
	unsigned tail;		//!< Last (smallest) element in deque
	unsigned head;		//!< First (largest) element in deque
	unsigned time;		//!< Current timestamp
};
struct tppm* tppm_init(unsigned chunk_size, unsigned max_age);
void tppm_free(struct tppm* tppm);
void tppm_process(struct tppm* tppm, const float* data, unsigned length);
