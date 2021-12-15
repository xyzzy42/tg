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

#if !GLIB_CHECK_VERSION(2,40,0)
static gboolean
g_key_file_save_to_file (GKeyFile     *key_file,
                         const gchar  *filename,
                         GError      **error)
{
  gchar *contents;
  gboolean success;
  gsize length;

  g_return_val_if_fail (key_file != NULL, FALSE);
  g_return_val_if_fail (filename != NULL, FALSE);
  g_return_val_if_fail (error == NULL || *error == NULL, FALSE);

  contents = g_key_file_to_data (key_file, &length, NULL);
  g_assert (contents != NULL);

  success = g_file_set_contents (filename, contents, length, error);
  g_free (contents);

  return success;
}
#endif

#define g_key_file_get_int g_key_file_get_integer
#define g_key_file_set_int g_key_file_set_integer // the devil may take glib
#define g_key_file_get_bool g_key_file_get_boolean
#define g_key_file_set_bool g_key_file_set_boolean

static void save_filter_chain(struct main_window *w)
{
	debug("Config: save filter chain\n");
	const unsigned int n = filter_chain_count(w->filter_chain);
	const struct biquad_filter *f = filter_chain_get(w->filter_chain, 0);

	if (!f || (f->type == HIGHPASS && n == 1)) {
		debug("Config: no chain, just save hpf frequency\n");
		/* Chain is a single HPF or empty, no chain needed */
		w->hpf_freq = f && f->enabled ? f->frequency : 0;
		g_key_file_remove_group(w->config_file, "filter_chain", NULL);
		return;
	}

	// g key files only support lists of a single type
	gint types[n], freqs[n];
	gdouble qs[n];
	gdouble gains[n];
	gboolean enabled[n];
	unsigned i;
	for (i = 0; i < n; i++) {
		f = filter_chain_get(w->filter_chain, i);
		types[i] = f->type;
		freqs[i] = f->frequency;
		qs[i] = f->bw;
		gains[i] = f->gain;
		enabled[i] = f->enabled;
	}
	debug("Config: saving chain with %u filters\n", n);
	g_key_file_set_integer_list(w->config_file, "filter_chain", "type", types, n);
	g_key_file_set_integer_list(w->config_file, "filter_chain", "frequency", freqs, n);
	g_key_file_set_double_list(w->config_file, "filter_chain", "q", qs, n);
	g_key_file_set_double_list(w->config_file, "filter_chain", "gain", gains, n);
	g_key_file_set_boolean_list(w->config_file, "filter_chain", "enabled", enabled, n);
}

static void load_filter_chain(struct main_window *w)
{
	struct filter_chain *chain = filter_chain_init(w->audio_rate ? (double)w->audio_rate : PA_SAMPLE_RATE);

	if (g_key_file_has_group(w->config_file, "filter_chain")) {
		gsize n, nn;
		g_autoptr(GError) e = NULL;
		g_autofree gint *types = NULL;
		g_autofree gint *freqs = NULL;
		g_autofree gdouble *qs = NULL;
		g_autofree gdouble *gains = NULL;
		g_autofree gboolean *enabled = NULL;

		types = g_key_file_get_integer_list(w->config_file, "filter_chain", "type", &n, &e);
		if (e) goto fail;
		freqs = g_key_file_get_integer_list(w->config_file, "filter_chain", "frequency", &nn, &e);
		if (e || n != nn) goto fail;
		qs = g_key_file_get_double_list(w->config_file, "filter_chain", "q", &nn, &e);
		if (e || n != nn) goto fail;
		gains = g_key_file_get_double_list(w->config_file, "filter_chain", "gain", &nn, &e);
		if (e || n != nn) goto fail;
		enabled = g_key_file_get_boolean_list(w->config_file, "filter_chain", "enabled", &nn, &e);
		if (e || n != nn) goto fail;

		debug("Config: Initial chain has %d filters\n", (int)n);
		unsigned i;
		for (i = 0; i < n; i++) {
			filter_chain_insert(chain, i);
			filter_chain_set(chain, i, types[i], freqs[i], qs[i], gains[i]);
		}
		for (i = 0; i < n; i++)
			filter_chain_enable(chain, i, enabled[i]);

		w->filter_chain = chain;
		return;

	fail:
		printf("Config: Error in filter chain configuration: %s\n",
		       e ? e->message : "Inconsistent number of filters");
		// Fall through to the "no filter_chain group" path and use a single HPF
	}

	// Just use highpass_cutoff_freq
	debug("Config: Use single HPF at %d\n", w->hpf_freq);
	filter_chain_insert(chain, 0);
	filter_chain_set(chain, 0, HIGHPASS, w->hpf_freq ? w->hpf_freq : FILTER_CUTOFF, M_SQRT1_2, 0);
	filter_chain_enable(chain, 0, w->hpf_freq != 0);
	w->filter_chain = chain;
	return;
}

void load_config(struct main_window *w)
{
	w->config_file = g_key_file_new();
	w->config_file_name = g_build_filename(g_get_user_config_dir(), CONFIG_FILE_NAME, NULL);
	w->conf_data = malloc(sizeof(struct conf_data));
#define SETUP(NAME,PLACE,TYPE) \
	w -> conf_data -> PLACE = w -> PLACE;

	CONFIG_FIELDS(SETUP);

	debug("Config: loading configuration file %s\n", w->config_file_name);
	gboolean ret = g_key_file_load_from_file(w->config_file, w->config_file_name, G_KEY_FILE_KEEP_COMMENTS, NULL);
	if(!ret) {
		debug("Config: failed to load config file");
		return;
	}
#define LOAD(NAME,PLACE,TYPE) \
	{ \
		GError *e = NULL; \
		TYPE val = g_key_file_get_ ## TYPE (w->config_file, "tg", #NAME, &e); \
		if(e) { \
			debug("Config: error loading field " #NAME "\n"); \
			g_error_free(e); \
		} else { \
			w -> PLACE = val; \
			w -> conf_data -> PLACE = val; \
		} \
	}

	CONFIG_FIELDS(LOAD);
	load_filter_chain(w);
}

void save_config(struct main_window *w)
{
	debug("Config: saving configuration file\n");

	g_key_file_set_string(w->config_file, "tg", "version", VERSION);

#define SAVE(NAME,PLACE,TYPE) \
	g_key_file_set_ ## TYPE (w->config_file, "tg", #NAME, w -> PLACE); \
	w -> conf_data -> PLACE = w -> PLACE;

	save_filter_chain(w);
	CONFIG_FIELDS(SAVE);

#ifdef DEBUG
	GError *ge = NULL;
	g_key_file_save_to_file(w->config_file, w->config_file_name, &ge);

	if(ge) {
		debug("Config: failed to save config file: %s\n",ge->message);
		g_error_free(ge);
		if(testing) exit(1);
	}
#else
	g_key_file_save_to_file(w->config_file, w->config_file_name, NULL);
#endif
}

void save_on_change(struct main_window *w)
{
#define CHANGED(NAME,PLACE,TYPE) \
	if(w -> PLACE != w -> conf_data -> PLACE) { \
		save_config(w); \
		return; \
	}

	CONFIG_FIELDS(CHANGED);
}

void close_config(struct main_window *w)
{
	g_key_file_free(w->config_file);
	g_free(w->config_file_name);
	free(w->conf_data);
}
