#include "tg.h"

/* Get rate from rate list combo box which has a manual entry.  Returns -1 on
 * error (entered rate not parsable). */
static int get_rate(GtkComboBox *rate_list)
{
	GtkTreeIter iter;
	unsigned int rate;

	if(gtk_combo_box_get_active_iter(rate_list, &iter)) {
		gtk_tree_model_get(gtk_combo_box_get_model(rate_list), &iter, 2, &rate, -1);
	} else {
		GtkEntry *entry = GTK_ENTRY(gtk_bin_get_child(GTK_BIN(rate_list)));
		if (sscanf(gtk_entry_get_text(entry), "%u", &rate) != 1)
			return -1;
		if (rate < 1000) { // Too slow, must mean kHz not Hz
			rate *= 1000;
		}
		char ratestr[16];
		snprintf(ratestr, sizeof(ratestr), "%u Hz", rate);
		gtk_entry_set_text(entry, ratestr);
	}
	return rate;
}

static void rate_entered(GtkWidget *rate_entry, struct main_window *w)
{
	UNUSED(rate_entry);
	gtk_range_set_fill_level(w->hpf_range, get_rate(w->rate_list) / 2);
}

static void rate_changed(GtkWidget *rate_list, struct main_window *w)
{
	/* Only act on combo list selections, not on every keystroke of manual
	 * entry.  rate_entered() will handle manual entry finished.  */
	if (gtk_combo_box_get_active(GTK_COMBO_BOX(rate_list)) == -1)
		return;
	gtk_range_set_fill_level(w->hpf_range, get_rate(w->rate_list) / 2);
}

static void populate_rate_list(GtkComboBox *rate_list, unsigned int current_rate, unsigned int rates)
{
	unsigned int i;
	GtkTreeIter iter;
	static const unsigned int freqs[] = AUDIO_RATES;
	static const char * const labels[] = AUDIO_RATE_LABELS;

	int active = -1; // -1 will set no active entry
	g_signal_handlers_block_matched(G_OBJECT(rate_list), G_SIGNAL_MATCH_FUNC, 0,0,NULL, rate_changed, NULL);
	GtkListStore *list = GTK_LIST_STORE(gtk_combo_box_get_model(rate_list));
	gtk_tree_model_get_iter_first(GTK_TREE_MODEL(list), &iter);
	for(i=0; i < ARRAY_SIZE(freqs); i++, gtk_tree_model_iter_next(GTK_TREE_MODEL(list), &iter)) {
		const bool supported = rates & (1u << i);
		gtk_list_store_set(list, &iter,
			0, labels[i],
			1, supported,
			2, freqs[i],
			-1);

		if (supported && current_rate == freqs[i])
			active = i;
	}
	g_signal_handlers_unblock_matched(G_OBJECT(rate_list), G_SIGNAL_MATCH_FUNC, 0,0,NULL, rate_changed, NULL);
	gtk_combo_box_set_active(rate_list, active);
	if (active == -1) {
		GtkWidget *entry = gtk_bin_get_child(GTK_BIN(rate_list));
		char ratestr[16];
		snprintf(ratestr, sizeof(ratestr), "%u Hz", current_rate);
		gtk_entry_set_text(GTK_ENTRY(entry), ratestr);
		gtk_widget_activate(entry);
	}
}

static void device_changed(GtkWidget *device_list, struct main_window *w)
{
	UNUSED(device_list);
	GtkTreeIter iter;
	unsigned int rates;

	if(!gtk_combo_box_get_active_iter(w->device_list, &iter))
		return;
	gtk_tree_model_get(gtk_combo_box_get_model(w->device_list), &iter, 2, &rates, -1);

	populate_rate_list(w->rate_list, w->nominal_sr, rates);
}

void audio_setup(GtkMenuItem *m, struct main_window *w)
{
	UNUSED(m);
	int i;

	const int current_dev = get_audio_device();
	const struct audio_device *devices;
	const int n = get_audio_devices(&devices);
	if (n < 0)  {
		error("Failed to get audio device list: %d\n", n);
		return;
	}
	// Populate list of devices
	g_signal_handlers_block_matched(G_OBJECT(w->device_list), G_SIGNAL_MATCH_DATA, 0,0,NULL,NULL, w);
	GtkListStore *list = GTK_LIST_STORE(gtk_combo_box_get_model(w->device_list));
	gtk_list_store_clear(list);
	for(i=0; i < n; i++)
		gtk_list_store_insert_with_values(list, NULL, -1,
			0, devices[i].name,
			1, devices[i].good,
			2, devices[i].rates,
			-1);
	g_signal_handlers_unblock_matched(G_OBJECT(w->device_list), G_SIGNAL_MATCH_DATA, 0,0,NULL,NULL, w);

	gtk_combo_box_set_active(w->device_list, current_dev);

	int response = gtk_dialog_run(GTK_DIALOG(w->audio_setup));
	gtk_widget_hide(w->audio_setup);
	if (response != GTK_RESPONSE_OK)
		return; // Cancel...

	int selected = gtk_combo_box_get_active(w->device_list);
	if (selected == -1)
		return; // Didn't select anything

	int new_rate = get_rate(w->rate_list);
	if (new_rate == -1)
		new_rate = w->nominal_sr; /* clear entry too? */

	int hpf_freq = gtk_range_get_value(w->hpf_range);

	i = set_audio_device(selected, &new_rate, NULL, hpf_freq, w->is_light);
	if (i == 0) {
		w->nominal_sr = new_rate;
		// Only save settings to config if it worked
		w->audio_rate = new_rate;
		w->hpf_freq = hpf_freq;
		// If selected dev is the default, save -1 "default" to config
		w->audio_device = devices[selected].isdefault ? -1 : selected;
		recompute(w);
	} else if (i < 0) {
		/* Try to restore old settings */
		new_rate = w->nominal_sr;
		i = set_audio_device(current_dev, &new_rate, NULL, w->hpf_freq, w->is_light);
		if (i < 0)
			error("Unable to restore previous audio settings.  Audio not working.");
	}
}

/* Setup and populate the audio setup widow */
void init_audio_dialog(struct main_window *w)
{
	w->audio_setup = gtk_dialog_new_with_buttons("Audio Setup", GTK_WINDOW(w->window),
		 GTK_DIALOG_DESTROY_WITH_PARENT,
		 "_Cancel", GTK_RESPONSE_CANCEL,
		 "_OK", GTK_RESPONSE_OK,
		 NULL);
	gtk_dialog_set_default_response(GTK_DIALOG(w->audio_setup), GTK_RESPONSE_OK);
	GtkWidget *content = gtk_dialog_get_content_area(GTK_DIALOG(w->audio_setup));

	GtkGrid *grid = GTK_GRID(gtk_grid_new());
	gtk_grid_set_row_spacing(grid, 12);
	gtk_grid_set_column_spacing(grid, 6);
	g_object_set(G_OBJECT(grid), "margin", 6, NULL);
	gtk_container_add(GTK_CONTAINER(content), GTK_WIDGET(grid));

	GtkWidget *label;

	gtk_grid_attach(grid, label = gtk_label_new("Audio Device"), 0, 0, 1, 1);
	g_object_set(G_OBJECT(label), "halign", GTK_ALIGN_END, NULL);

	GtkCellRenderer *renderer;
	GtkListStore *list;

	/* 0 Name; 1 suitable for recording; 2 bitmask of allowed rates */
	list = gtk_list_store_new(3, G_TYPE_STRING, G_TYPE_BOOLEAN, G_TYPE_UINT);
	GtkWidget *devices = gtk_combo_box_new_with_model(GTK_TREE_MODEL(list));
	g_object_unref(list);
	w->device_list = GTK_COMBO_BOX(devices);
	gtk_combo_box_set_active(w->device_list, get_audio_device());
	renderer = gtk_cell_renderer_text_new();
	gtk_cell_layout_pack_start(GTK_CELL_LAYOUT(devices), renderer, TRUE);
	gtk_cell_layout_set_attributes(GTK_CELL_LAYOUT(devices), renderer, "text", 0, "sensitive", 1, NULL);
	gtk_widget_set_hexpand(devices, true); // One cell with hexpand affects the entire column
	gtk_grid_attach(grid, devices, 1, 0, 1, 1);

	gtk_grid_attach(grid, label = gtk_label_new("Sample Rate"), 0, 1, 1, 1);
	g_object_set(G_OBJECT(label), "halign", GTK_ALIGN_END, NULL);

	/* 0 Name; 1 Supported rate?; 2 integer rate in Hz */
	list = gtk_list_store_new(3, G_TYPE_STRING, G_TYPE_BOOLEAN, G_TYPE_UINT);
	int n;
	GtkTreeIter iter;
	for (n = NUM_AUDIO_RATES; n; n--) gtk_list_store_insert(list, &iter, -1);
	GtkWidget *rates = gtk_combo_box_new_with_model_and_entry(GTK_TREE_MODEL(list));
	g_object_unref(list);
	w->rate_list = GTK_COMBO_BOX(rates);
	gtk_combo_box_set_entry_text_column(GTK_COMBO_BOX(rates), 0);
	renderer = gtk_cell_renderer_text_new();
	gtk_cell_layout_clear(GTK_CELL_LAYOUT(rates));
	gtk_cell_layout_pack_start(GTK_CELL_LAYOUT(rates), renderer, TRUE);
	gtk_cell_layout_set_attributes(GTK_CELL_LAYOUT(rates), renderer, "text", 0, "sensitive", 1, NULL);
	gtk_grid_attach(grid, rates, 1, 1, 1, 1);

	gtk_grid_attach(grid, label = gtk_label_new("High Pass Cutoff"), 0, 2, 1, 1);
	g_object_set(G_OBJECT(label), "halign", GTK_ALIGN_END, NULL);

	GtkWidget *hpf = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0, 24000, 100);
	w->hpf_range = GTK_RANGE(hpf);
	gtk_scale_add_mark(GTK_SCALE(w->hpf_range), 0, GTK_POS_BOTTOM, "Off");
	gtk_scale_add_mark(GTK_SCALE(w->hpf_range), FILTER_CUTOFF, GTK_POS_BOTTOM, "Default");
	gtk_range_set_restrict_to_fill_level(w->hpf_range, true);
	gtk_range_set_show_fill_level(w->hpf_range, true);
	gtk_range_set_value(w->hpf_range, FILTER_CUTOFF);
	gtk_grid_attach(grid, hpf, 1, 2, 1, 1);

	gtk_widget_show_all(content);

	g_signal_connect(G_OBJECT(w->device_list), "changed", G_CALLBACK(device_changed), w);
	g_signal_connect(G_OBJECT(gtk_bin_get_child(GTK_BIN(w->rate_list))), "activate", G_CALLBACK(rate_entered), w);
	g_signal_connect(G_OBJECT(w->rate_list), "changed", G_CALLBACK(rate_changed), w);
}
