#include "tg.h"

#ifdef DEBUG
#define prdbg(fmt, ...) printf("%s: " fmt "\n", __func__, ## __VA_ARGS__)
#else
__attribute__((format(printf, 1, 2)))
static inline void prdbg(const char *fmt, ...)
{ (void)fmt; }
#endif

#define box_pack_start(box, widget, expand, fill, padding) \
	gtk_box_pack_start(GTK_BOX((box)), GTK_WIDGET((widget)), (expand), (fill), (padding))
#define box_pack_end(box, widget, expand, fill, padding) \
	gtk_box_pack_end(GTK_BOX((box)), GTK_WIDGET((widget)), (expand), (fill), (padding))
#define container_add(container, widget) \
	gtk_container_add(GTK_CONTAINER((container)), GTK_WIDGET((widget)))


//  Spin Slider widget
typedef struct {
	GtkSpinButton*	spin;
	GtkLabel*	label;
	GtkBox*		box;
} SpinSliderPrivate;

typedef struct {
	GtkScale	parent;
} SpinSlider;

typedef struct {
	GtkScaleClass	parent_class;
} SpinSliderClass;

G_DEFINE_TYPE_WITH_CODE(SpinSlider, spin_slider, GTK_TYPE_SCALE,
	G_ADD_PRIVATE(SpinSlider)	// This adds private data named SpinSliderPrivate
	)

#define SPIN_SLIDER_TYPE	(spin_slider_get_type())
#define SPIN_SLIDER(obj)	(G_TYPE_CHECK_INSTANCE_CAST((obj), SPIN_SLIDER_TYPE, SpinSlider))

static inline SpinSliderPrivate* spin_slider_get_priv(SpinSlider *ss)
{ return spin_slider_get_instance_private(ss); }

GtkBox*
spin_slider_get_container(SpinSlider *ss)
{
	return spin_slider_get_priv(ss)->box;
}

static void
spin_slider_set_property (GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
	SpinSliderPrivate *priv = spin_slider_get_instance_private(SPIN_SLIDER(object));
	prdbg("set property %s %u of %p", pspec->name, prop_id, priv);

	switch (prop_id) {
	case 1:
		g_object_set_property(G_OBJECT(priv->label), pspec->name, value);
		break;
	case 2:
		g_object_set_property(G_OBJECT(priv->box), pspec->name, value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}
}

static void
spin_slider_get_property (GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
	SpinSliderPrivate *priv = spin_slider_get_instance_private(SPIN_SLIDER(object));
	prdbg("get property %s %u of %p", pspec->name, prop_id, priv);

	switch (prop_id) {
	case 1:
		g_object_get_property(G_OBJECT(priv->label), pspec->name, value);
		break;
	case 2:
		g_object_get_property(G_OBJECT(priv->box), pspec->name, value);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
		break;
	}
}

// Initialize each object
static void spin_slider_init(SpinSlider* ss)
{
	prdbg("start %p", ss);
	GtkRange *range = GTK_RANGE(&ss->parent); // Parent object
	prdbg("range is %p", range);
	SpinSliderPrivate *priv = spin_slider_get_instance_private(ss);
	prdbg("private pointer at %p", priv);

	// Parameters from _new are not set yet

	// Create child objects
	priv->box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10));
	g_object_ref_sink(priv->box);

	GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	priv->label = GTK_LABEL(gtk_label_new(""));
	box_pack_start(priv->box, vbox, FALSE, FALSE, 0);
	box_pack_start(vbox, priv->label, FALSE, FALSE, 0);
	// Doesn't seem to propagate ss -> label, but does label -> ss ???
	//g_object_bind_property(G_OBJECT(ss), "label", G_OBJECT(priv->label), "label", G_BINDING_BIDIRECTIONAL);
	priv->spin = GTK_SPIN_BUTTON(gtk_spin_button_new_with_range(0, 1, 1));
	box_pack_start(vbox, priv->spin, FALSE, FALSE, 0);
	gtk_widget_set_valign(GTK_WIDGET(priv->spin), GTK_ALIGN_CENTER);

	gtk_range_set_restrict_to_fill_level(range, true);
	gtk_range_set_show_fill_level(range, true);
	box_pack_start(priv->box, range, TRUE, TRUE, 0);
	g_object_bind_property(gtk_range_get_adjustment(range), "value", priv->spin, "value", G_BINDING_BIDIRECTIONAL);
	g_object_bind_property(range, "adjustment", priv->spin, "adjustment", G_BINDING_DEFAULT);
	g_object_bind_property(range, "digits", priv->spin, "digits", G_BINDING_DEFAULT);
	prdbg("done");
}

static void spin_slider_dispose(GObject* object)
{
	prdbg("dispose %p", object);
	SpinSliderPrivate *priv = spin_slider_get_instance_private(SPIN_SLIDER(object));
	if (priv->box) {
		g_object_unref(priv->box);
		priv->box = NULL;
	}
	G_OBJECT_CLASS(spin_slider_parent_class)->dispose(object);
	prdbg("done");
}

// Class init function
static void spin_slider_class_init(SpinSliderClass *klass)
{
	G_OBJECT_CLASS(klass)->set_property = spin_slider_set_property;
	G_OBJECT_CLASS(klass)->get_property = spin_slider_get_property;
	G_OBJECT_CLASS(klass)->dispose = spin_slider_dispose;

	g_object_class_install_property(G_OBJECT_CLASS(klass),
		1,
		g_param_spec_string("label", "label", "label", NULL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_EXPLICIT_NOTIFY));

	g_object_class_override_property(G_OBJECT_CLASS(klass), 2, "sensitive");
}

GtkWidget* spin_slider_new(const char* label, gdouble min, gdouble max, gdouble step)
{
	GtkAdjustment *adj = gtk_adjustment_new(min, min, max, step, 10*step, 0);
	gint digits = (fabs(step) >= 1.0 || step == 0.0) ? 0 : abs((int)floor(log10(fabs(step))));
	SpinSlider* ss = g_object_new(SPIN_SLIDER_TYPE,
		"label", label,
		"adjustment", adj,
		"digits", digits,
		NULL);
	prdbg("object created");
	SpinSliderPrivate *priv = spin_slider_get_instance_private(ss);
	gtk_widget_set_vexpand(GTK_WIDGET(priv->spin), FALSE); // doesn't work?
	return GTK_WIDGET(ss);
}

// Filter Dialog Widget
static const struct {
	double	q;
	unsigned freq;
	bool hasgain;
	bool hasq;
} filter_defaults[NUM_BITYPES] = {
	[NOTCH] =    { 0.100,     22000, false, false },
	[BANDPASS] = { 0.150,     15000, false, false },
	[LOWPASS] =  { M_SQRT1_2, 23000, false, true },
	[HIGHPASS] = { M_SQRT1_2,  3000, false, true },
	[ALLPASS] =  { 0.167,      7800, false, false },
	[PEAK] =     { 0.228,     15000, true,  false },
};

static const char* const filter_names[NUM_BITYPES] = {
	[HIGHPASS] = "High Pass",
	[LOWPASS] = "Low Pass",
	[BANDPASS] = "Band Pass",
	[NOTCH] = "Notch",
	[ALLPASS] = "All Pass",
	[PEAK] = "Peak",
};

typedef struct _FilterDialog {
	GtkDialog parent;
} FilterDialog;

typedef struct _FilterDialogClass {
	GtkDialogClass parent_class;
} FilterDialogClass;

typedef struct _FilterDialogPrivate {
	/** Filter List Pane */
	/// @{
	GtkTreeModel*	model;	//< List of filters
	GtkTreeView*	view;	//< TreeView of the filter list above
	GtkComboBox*	newtype;//< New filter type for add button
	GtkTreePath*	selection; //< Last selection
	struct filter_chain* chain; //< Filter chain, should match model
	/// @}

	int	dnd_src;	//< Drag & Drop source row when DND is active
	int	dnd_dst;	//< Drag & Drop desination row when DND is active

	/** Filter Editing Pane */
	/// @{
	GtkWidget*		editframe;
	GtkComboBox*		type;
	GtkToggleButton*	enabled;
	GtkRange*		center;
	GtkRange*		q;
	GtkRange*		gain;
	bool			edit_updated;
	/// @}

	/** Filter response graph */
	/// @{
	GtkImage*		image;
	bool			dograph;
	bool			fullchain;
	bool			graph_updated;
	gint			graph_id;
	/// @}
} FilterDialogPrivate;

#define FILTER_DIALOG_TYPE (filter_dialog_get_type())
#define FILTER_DIALOG(o) (G_TYPE_CHECK_INSTANCE_CAST ((o), FILTER_DIALOG_TYPE, FilterDialog))

G_DEFINE_TYPE_WITH_PRIVATE(FilterDialog, filter_dialog, GTK_TYPE_DIALOG);

enum _filter_list_columns {
	COL_TYPE,
	COL_FREQ,
	COL_Q,
	COL_GAIN,
	COL_ENABLED,
	COL_ID,
	COL_NONBLANK,
	COL_MAX
};

void filter_dialog_set_chain(FilterDialog* filter_dialog, struct filter_chain* chain);
static void filter_edit_check(FilterDialogPrivate *priv, const gchar* path, unsigned index);

GtkWidget* filter_dialog_new(struct main_window *w)
{
	FilterDialog* f = g_object_new(FILTER_DIALOG_TYPE,
		"title", "Audio Filter Chain",
		"use-header-bar", FALSE,
		"application", w->app,
		"destroy-with-parent", TRUE,
		"transient-for", GTK_WINDOW(w->window),
		NULL);
	filter_dialog_set_chain(f, w->filter_chain);
	return GTK_WIDGET(f);
}

static void filter_dialog_dispose(GObject* object)
{
	prdbg("dispose %p", object);
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(FILTER_DIALOG(object));
	prdbg("priv is %p", priv);
	if (priv->model) {
		g_object_unref(priv->model);
		priv->model = NULL;
	}
	G_OBJECT_CLASS(filter_dialog_parent_class)->dispose(object);
	prdbg("done");
}

static void filter_dialog_filter_changed(FilterDialog* filter_dialog, guint index)
{
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(FILTER_DIALOG(filter_dialog));
	const struct biquad_filter* filter = filter_chain_get(priv->chain, index);
	prdbg("filter %u changed in %p", index, priv);
	if (!filter)
		return;
	GtkTreeIter iter;
	if (!gtk_tree_model_iter_nth_child(priv->model, &iter, NULL, index))
		return;

	filter_edit_check(priv, NULL, index);
	// This will trigger the filter list store's change callback, which will try to program
	// this back into the chain, but that's ok, since setting a filter to its existing
	// values is a no-op.  So we don't have to block the callback.
	gtk_list_store_set(GTK_LIST_STORE(priv->model), &iter,
		COL_TYPE, filter->type,
		COL_Q, filter->bw,
		COL_FREQ, filter->frequency,
		COL_ENABLED, filter->enabled,
		-1);
}

static void filter_dialog_class_init(FilterDialogClass* class)
{
	prdbg("class init");
	G_OBJECT_CLASS(class)->dispose = filter_dialog_dispose;

	g_signal_new_class_handler("filter-changed", FILTER_DIALOG_TYPE,
		G_SIGNAL_RUN_LAST | G_SIGNAL_ACTION,
		G_CALLBACK(filter_dialog_filter_changed),
		NULL, NULL,
		g_cclosure_marshal_VOID__UINT,
		G_TYPE_NONE, 1, G_TYPE_UINT);
}

static GtkToolItem* create_tool_button(const char *name, GCallback func, void *arg)
{
	GtkToolItem *item = gtk_tool_button_new(NULL, NULL);
	gtk_tool_button_set_icon_name(GTK_TOOL_BUTTON(item), name);
	if (func != NULL)
		g_signal_connect(G_OBJECT(item), "clicked", func, arg);
	return item;
}

static void add_filter(GtkButton* button, FilterDialog* filter_dialog)
{
	UNUSED(button);
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(filter_dialog);

	const gchar* typestr = gtk_combo_box_get_active_id(priv->newtype);
	if (!typestr)
		return; // No type selected?
	unsigned typeid = strtoul(typestr, NULL, 0);

	const guint freq = filter_defaults[typeid].freq;
	const gdouble q = filter_defaults[typeid].q;
	const gint n = gtk_tree_model_iter_n_children(priv->model, NULL);
	prdbg("add filter #%d: %u %u %f", n, typeid, freq, q);
	GtkTreeIter iter;
	gtk_list_store_insert_with_values(GTK_LIST_STORE(priv->model), &iter, n,
		COL_TYPE, typeid,
		COL_ENABLED, FALSE,
		COL_ID, n,
		COL_NONBLANK, TRUE,
		COL_FREQ, freq,
		COL_Q, q,
		-1);
	g_autoptr(GtkTreePath) path = gtk_tree_model_get_path(priv->model, &iter);
	gtk_tree_view_set_cursor(priv->view, path, NULL, false);

	gint *indices = gtk_tree_path_get_indices(path);
	prdbg("added new filter #%d", indices[0]);
}

static void remove_filter(GtkButton* button, FilterDialog* filter_dialog)
{
	UNUSED(button);
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(filter_dialog);
	prdbg("filter remove clicked");

	GtkTreeIter iter;
	if (!gtk_tree_selection_get_selected(gtk_tree_view_get_selection(priv->view), NULL, &iter))
		return;
	gtk_tree_path_free(priv->selection);
	priv->selection = NULL;
	prdbg("remove from list");
	gtk_list_store_remove(GTK_LIST_STORE(priv->model), &iter);
}

static GtkTreeView *create_listview(FilterDialogPrivate* priv);
static void filter_settings_update(FilterDialogPrivate *priv, GtkTreeIter* iter, bool audio);
static void filter_graph_update(FilterDialogPrivate *priv, unsigned id);

static int filter_selection_to_row(FilterDialogPrivate* priv)
{
	if (!priv->selection)
		return -1;
	return gtk_tree_path_get_indices(priv->selection)[0];
}

// Handle any changes to the controls in the filter edit pane
static void filter_edit_apply(GtkRange *range, FilterDialogPrivate* priv)
{
	UNUSED(range);
	prdbg("apply sliders into store");

	// Get iter to selected filter
	GtkTreeSelection *sel = gtk_tree_view_get_selection(priv->view);
	if (!sel)
		return;
	GtkTreeIter iter;
	gtk_tree_selection_get_selected(sel, NULL, &iter);

	// Get values from Edit Filter area
	const gdouble q = gtk_range_get_value(priv->q);
	const guint freq = gtk_range_get_value(priv->center);
	const gdouble gain = gtk_range_get_value(priv->gain);
	const gchar* typestr = gtk_combo_box_get_active_id(priv->type);
	const gboolean enabled = gtk_toggle_button_get_active(priv->enabled);
	if (!typestr)
		return; // No type selected?
	unsigned typeid = strtoul(typestr, NULL, 0);

	prdbg("update filter to %u %u %f", typeid, freq, q);
	gtk_list_store_set(GTK_LIST_STORE(priv->model), &iter,
		COL_TYPE, typeid,
		COL_Q, q,
		COL_FREQ, freq,
		COL_GAIN, gain,
		COL_ENABLED, enabled,
		-1);
}

static void filter_graph_toggled(GtkToggleButton *button, FilterDialogPrivate* priv)
{
	priv->dograph = gtk_toggle_button_get_active(button);
	if (priv->dograph) {
		priv->graph_updated = false;
		filter_graph_update(priv, filter_selection_to_row(priv));
	} else {
		priv->graph_updated = true;
		gtk_image_clear(priv->image);
	}
}

static void filter_graph_source(GtkToggleButton *button, FilterDialogPrivate* priv)
{
	if (gtk_toggle_button_get_active(button)) {
		bool fullchain = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(button), "id")) == 1;
		prdbg("set to %s", fullchain ? "fullchain" : "selected filter");
		if (priv->fullchain != fullchain) {
			priv->fullchain = fullchain;
			priv->graph_updated = false;
			filter_graph_update(priv, filter_selection_to_row(priv));
		}
	}
}

static void dialog_response(GtkDialog *dialog, int response_id, gpointer unused)
{
	UNUSED(unused);
	if (response_id == GTK_RESPONSE_CLOSE)
		gtk_widget_hide(GTK_WIDGET(dialog));
}

static gboolean active_to_sensitive(GBinding* binding, const GValue *src, GValue* dst, gpointer user_data)
{
	UNUSED(binding);
	UNUSED(user_data);
	g_value_set_boolean(dst, g_value_get_int(src) >= 0);
	return TRUE;
}

static void add_filter_choices(GtkComboBoxText* cb)
{
	unsigned i;
	for (i = 0; i < NUM_BITYPES; i++) {
		if (filter_names[i]) {
			char id[3];
			snprintf(id, sizeof(id), "%u", i);
			gtk_combo_box_text_append(cb, id, filter_names[i]);
		}
	}
}


static void filter_list_changed(GtkTreeModel* model, GtkTreePath* path, GtkTreeIter* iter, FilterDialogPrivate *priv);
static void filter_list_deleted(GtkTreeModel* model, GtkTreePath* path, FilterDialogPrivate *priv);
static void filter_list_inserted(GtkTreeModel* model, GtkTreePath* path, GtkTreeIter* iter, FilterDialogPrivate *priv);

static void filter_dialog_init(FilterDialog* filter_dialog)
{
	GtkDialog *dialog = &filter_dialog->parent;
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(filter_dialog);

	/* Initial values */
	*priv = (FilterDialogPrivate){
		.model = NULL,
		.view = NULL,
		.dnd_src = -1,
		.dnd_dst = -1,
	};

	gtk_dialog_add_buttons(dialog,
		 "_Close", GTK_RESPONSE_CLOSE,
		 NULL);
	g_signal_connect(G_OBJECT(dialog), "response", G_CALLBACK(dialog_response), NULL);
	g_signal_connect(G_OBJECT(dialog), "delete-event", G_CALLBACK(gtk_widget_hide_on_delete), NULL);

	GtkWidget *paned = gtk_paned_new(GTK_ORIENTATION_VERTICAL);
	gtk_paned_set_wide_handle(GTK_PANED(paned), TRUE);
	container_add(gtk_dialog_get_content_area(dialog), paned);

	// Filter List Area
	//
	GtkBox *lbox = GTK_BOX(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
	gtk_paned_pack1(GTK_PANED(paned), GTK_WIDGET(lbox), TRUE, FALSE);
	GtkWidget *scrolled = gtk_scrolled_window_new(NULL, NULL);
	box_pack_start(lbox, scrolled, TRUE, TRUE, 0);
	gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled), GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
	priv->model = GTK_TREE_MODEL(gtk_list_store_new(COL_MAX,
		G_TYPE_UINT,	// Type
		G_TYPE_UINT, 	// Frequency Center
		G_TYPE_DOUBLE,	// Q or BW
		G_TYPE_DOUBLE,	// Gain
		G_TYPE_BOOLEAN,	// Enabled
		G_TYPE_INT,	// Filter #
		G_TYPE_BOOLEAN	// Non-blank flag
	));
	g_signal_connect(priv->model, "row-changed", G_CALLBACK(filter_list_changed), priv);
	g_signal_connect(priv->model, "row-deleted", G_CALLBACK(filter_list_deleted), priv);
	g_signal_connect(priv->model, "row-inserted", G_CALLBACK(filter_list_inserted), priv);
	priv->view = create_listview(priv);
	container_add(scrolled, priv->view);

	// Toolbar
	GtkToolbar *toolbar = GTK_TOOLBAR(gtk_toolbar_new());
	box_pack_start(lbox, toolbar, FALSE, FALSE, 0);
	gtk_toolbar_set_icon_size(toolbar, GTK_ICON_SIZE_BUTTON);
	GtkStyleContext *context = gtk_widget_get_style_context(GTK_WIDGET(toolbar));
	gtk_style_context_add_class(context, GTK_STYLE_CLASS_INLINE_TOOLBAR);

	// Toolbar new type combo box
	GtkToolItem* toolitem = gtk_tool_item_new();
	gtk_toolbar_insert(toolbar, toolitem, -1);
	GtkWidget* nbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
	container_add(toolitem, nbox);
	box_pack_start(nbox, gtk_label_new("New Filter Type:"), FALSE, FALSE, 0);
	priv->newtype = GTK_COMBO_BOX(gtk_combo_box_text_new());
	box_pack_start(nbox, priv->newtype, FALSE, FALSE, 0);
	add_filter_choices(GTK_COMBO_BOX_TEXT(priv->newtype));

	// Toolbar buttons
	GtkToolItem *tb = create_tool_button("list-add", G_CALLBACK(add_filter), filter_dialog);
	g_object_bind_property_full(priv->newtype, "active", tb, "sensitive",
		G_BINDING_DEFAULT | G_BINDING_SYNC_CREATE,
		active_to_sensitive, NULL, NULL, NULL);
	gtk_toolbar_insert(toolbar, tb, -1);
	gtk_toolbar_insert(toolbar, create_tool_button("list-remove", G_CALLBACK(remove_filter), filter_dialog), -1);

	// Filter Edit Area
	//
	GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
	gtk_paned_pack2(GTK_PANED(paned), GTK_WIDGET(vbox), TRUE, TRUE);
	priv->editframe = gtk_frame_new("Edit Filter");
	box_pack_start(vbox, priv->editframe, FALSE, FALSE, 0);
	GtkWidget *ebox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	container_add(priv->editframe, ebox);
	gtk_box_set_homogeneous(GTK_BOX(ebox), TRUE);

	// Filter Type + Enabled
	GtkWidget *hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
	box_pack_end(ebox, hbox, FALSE, FALSE, 0);

	// Filter Type
	box_pack_start(hbox, gtk_label_new("Filter Type:"), FALSE, FALSE, 0);
	priv->type = GTK_COMBO_BOX(gtk_combo_box_text_new());
	box_pack_start(hbox, priv->type, FALSE, FALSE, 0);
	add_filter_choices(GTK_COMBO_BOX_TEXT(priv->type));
	g_signal_connect(G_OBJECT(priv->type), "changed", G_CALLBACK(filter_edit_apply), priv);

	// Enabled
	priv->enabled = GTK_TOGGLE_BUTTON(gtk_check_button_new_with_label("Enabled"));
	box_pack_end(hbox, priv->enabled, FALSE, FALSE, 0);
	g_signal_connect(G_OBJECT(priv->enabled), "toggled", G_CALLBACK(filter_edit_apply), priv);

	// Frequency control
	SpinSlider *ss = SPIN_SLIDER(spin_slider_new("Frequency Center:", 0, 24000, 100));
	priv->center = GTK_RANGE(ss);
	box_pack_start(ebox, spin_slider_get_container(ss), FALSE, FALSE, 0);
	g_signal_connect(G_OBJECT(ss), "value-changed", G_CALLBACK(filter_edit_apply), priv);

	// Q control
	ss = SPIN_SLIDER(spin_slider_new("Q / BW:", 0, 1, .001));
	priv->q = GTK_RANGE(ss);
	box_pack_start(ebox, spin_slider_get_container(ss), FALSE, FALSE, 0);
	gtk_scale_add_mark(GTK_SCALE(priv->q), 0, GTK_POS_BOTTOM, "Default");
	g_signal_connect(G_OBJECT(ss), "value-changed", G_CALLBACK(filter_edit_apply), priv);

	// Gain (only some filters have this)
	ss = SPIN_SLIDER(spin_slider_new("Gain", -35, 35, .01));
	priv->gain = GTK_RANGE(ss);
	box_pack_start(ebox, spin_slider_get_container(ss), FALSE, FALSE, 0);
	gtk_scale_add_mark(GTK_SCALE(priv->gain), 0, GTK_POS_BOTTOM, "");
	g_signal_connect(G_OBJECT(ss), "value-changed", G_CALLBACK(filter_edit_apply), priv);

	// Filter Graph Area
	GtkWidget *frame = gtk_frame_new("Filter Response");
	box_pack_start(vbox, frame, TRUE, TRUE, 0);
	GtkWidget *button = gtk_check_button_new_with_label("Filter Response Graph");
	gtk_frame_set_label_widget(GTK_FRAME(frame), button);
	g_signal_connect(G_OBJECT(button), "toggled", G_CALLBACK(filter_graph_toggled), priv);
	ebox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	container_add(frame, ebox);

	// Graph Srouce
	hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
	box_pack_start(ebox, hbox, FALSE, FALSE, 0);
	box_pack_start(hbox, gtk_label_new("Graph source:"), FALSE, FALSE, 0);
	GtkWidget *single = gtk_radio_button_new_with_label(NULL, "Selected Filter Only");
	box_pack_start(hbox, single, FALSE, FALSE, 0);
	g_signal_connect(G_OBJECT(single), "toggled", G_CALLBACK(filter_graph_source), priv);
	g_object_set_data(G_OBJECT(single), "id", GUINT_TO_POINTER(0));
	GtkWidget *all = gtk_radio_button_new_with_label_from_widget(GTK_RADIO_BUTTON(single), "Entire Chain");
	box_pack_start(hbox, all, FALSE, FALSE, 0);
	g_signal_connect(G_OBJECT(all), "toggled", G_CALLBACK(filter_graph_source), priv);
	g_object_set_data(G_OBJECT(all), "id", GUINT_TO_POINTER(1));

	// Image
	priv->image = GTK_IMAGE(gtk_image_new());
	image_set_minimum_size(priv->image, 640, 480);
	box_pack_start(ebox, priv->image, TRUE, TRUE, 0);
	gtk_widget_set_vexpand(GTK_WIDGET(priv->image), TRUE);
	gtk_widget_set_hexpand(GTK_WIDGET(priv->image), TRUE);
	gtk_widget_set_size_request(GTK_WIDGET(priv->image), 640, -1);
}

/* Update IDs from [a, b] to be correct */
static void filter_list_fixup(FilterDialogPrivate* priv, int a, int b)
{
	int i = a < b ? a : b;
	int end = a < b ? b : a;

	g_signal_handlers_block_matched(G_OBJECT(priv->model), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_list_changed, NULL);

	if (priv->selection) {
		const gint *indices = gtk_tree_path_get_indices(priv->selection);
		if (indices && indices[0] >= i && indices[0] <= end) {
			prdbg("clearing selection %u in range [%d, %d]", indices[0], i, end);
			/* We could adjust the path, but just clearing it should be enough */
			gtk_tree_path_free(priv->selection);
			priv->selection = NULL;
		}
	}

	GtkTreeIter iter;
	gtk_tree_model_iter_nth_child(priv->model, &iter, NULL, i);
	while (i <= end) {
		prdbg("Update ID of new row %d", i);
		gtk_list_store_set(GTK_LIST_STORE(priv->model), &iter, COL_ID, i++, -1);
		gtk_tree_model_iter_next(priv->model, &iter);
	}

	g_signal_handlers_unblock_matched(G_OBJECT(priv->model), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_list_changed, NULL);
}

// Update the sliders and/or graph and/or audio system when the data in the
// model changes or the selected filter changes.
// Uses private state vales edit/graph_updated to determine if they need to be refreshed.
static void filter_settings_update(FilterDialogPrivate *priv, GtkTreeIter* iter, bool audio)
{
	guint freq, id;
	gdouble q, gain;
	enum bitype type;
	gboolean enabled;
	gtk_tree_model_get(priv->model, iter,
		COL_TYPE, &type, COL_FREQ, &freq, COL_Q, &q, COL_GAIN, &gain, COL_ID, &id, COL_ENABLED, &enabled,-1);
	prdbg("Update filter #%u: %u %u %f", id, type, freq, q);

	if (!priv->edit_updated) {
		prdbg("update sliders");
		g_signal_handlers_block_matched(G_OBJECT(priv->center), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_block_matched(G_OBJECT(priv->q), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_block_matched(G_OBJECT(priv->gain), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_block_matched(G_OBJECT(priv->type), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_block_matched(G_OBJECT(priv->enabled), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);

		gtk_range_set_range(priv->q, 0, filter_defaults[type].hasq ? 10 : 1);
		gtk_combo_box_set_active(priv->type, type);
		gtk_range_set_value(priv->center, freq);
		gtk_range_set_value(priv->q, q);
		gtk_range_set_value(priv->gain, gain);
		g_object_set(priv->q, "label", filter_defaults[type].hasq ? "Q" : "BW", NULL);
		g_object_set(priv->gain, "sensitive", filter_defaults[type].hasgain, NULL);
		gtk_toggle_button_set_active(priv->enabled, enabled);
		gtk_scale_clear_marks(GTK_SCALE(priv->q));
		if (filter_defaults[type].q > 0)
			gtk_scale_add_mark(GTK_SCALE(priv->q), filter_defaults[type].q, GTK_POS_BOTTOM, "Default");

		g_signal_handlers_unblock_matched(G_OBJECT(priv->center), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_unblock_matched(G_OBJECT(priv->q), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_unblock_matched(G_OBJECT(priv->gain), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_unblock_matched(G_OBJECT(priv->type), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		g_signal_handlers_unblock_matched(G_OBJECT(priv->enabled), G_SIGNAL_MATCH_FUNC, 0,0,NULL, filter_edit_apply, NULL);
		priv->edit_updated = true;
	}

	if (audio) {
		prdbg("update audio system filter");
		if (!filter_chain_set(priv->chain, id, type, freq, q, gain))
			prdbg("unneeded audio system update");
		filter_chain_enable(priv->chain, id, enabled);
	}

	filter_graph_update(priv, id);
}

static gboolean do_graph(gpointer user_data)
{
	FilterDialogPrivate *priv = user_data;
	if (!priv->graph_updated && priv->dograph) {
		prdbg("Plot filter to %p", priv->image);
		if (priv->fullchain)
			create_filter_chain_plot(priv->image);
		else {
			create_filter_n_plot(priv->image, priv->graph_id);
		}
		priv->graph_updated = true;
	}
	return FALSE;
}

static void filter_graph_update(FilterDialogPrivate *priv, unsigned id)
{
	if (!priv->graph_updated && priv->dograph) {
		prdbg("schedule filter plot");
		priv->graph_id = id;
		g_idle_add(do_graph, priv);
	}
}

/* Does a change to a filter mean the graph must update?  Sets priv->graph_updated to
 * false if it does.  path is filter changed and enabled is its enabled status.  The
 * graph is invalidated if an enabled filter changes in fullchain mode and if the
 * selected filter changes in selected filter mode.  */
static void filter_check_graph(FilterDialogPrivate *priv, GtkTreePath* path, gboolean enabled)
{
	prdbg("check graph for change to %senabled filter #%u", enabled?"":"un", gtk_tree_path_get_indices(path)[0]);
	// Did this invalidate graph?
	if (priv->fullchain) {
		if (enabled)
			priv->graph_updated = false;
	} else {
		// Did we get a change to selected filter?
		GtkTreeSelection *sel = gtk_tree_view_get_selection(priv->view);
		GtkTreeIter iter;
		if (gtk_tree_selection_get_selected(sel, NULL, &iter)) {
			g_autoptr(GtkTreePath) spath = gtk_tree_model_get_path(priv->model, &iter);
			prdbg("compare selected path %p to changed filter %p", spath, path);
			if (gtk_tree_path_compare(spath, path) == 0)
				priv->graph_updated = false;
		}
	}
	if (!priv->graph_updated) prdbg("graph needs update");
}

/* Data in the model was changed.  This is the "real" data.  Various UI elements
 * show it, but their values aren't used as data.  Then those UI elements change
 * their signals push data into the model, which in turn signals this callback.
 * */
static void filter_list_changed(GtkTreeModel* model, GtkTreePath* path, GtkTreeIter* iter, FilterDialogPrivate *priv)
{
	gint id, row = gtk_tree_path_get_indices(path)[0];
	gboolean enabled, nonblank;
	gtk_tree_model_get(model, iter, COL_ID, &id, COL_NONBLANK, &nonblank, COL_ENABLED, &enabled, -1);
	prdbg("changed %sblank filter, id is %u at row %d", nonblank ? "non-" : "", id, row);

	/* There are two ways this gets called by gtk.  One is when the data changes.
	 * The other the middle part of a three step drag&drop process.  In the latter
	 * case, we make a note of the ongoing DND which is handled when the third and
	 * final DND step triggers the "deleted" callback.  */
	if (priv->dnd_dst != -1) {
		if (priv->dnd_dst != row) {
			printf("DND in progress but non-dropped row is getting changed?\n");
			/* handle as non-dnd? Abort dnd? */
			return;
		}
		priv->dnd_src = id;
		prdbg("DND in progress, changed row %d to have original row %d", row, id);
		/* Handle after delete goes through */
	} else {
		if (id != row)
			printf("DND not in progress but row has incorrect ID? %u in row %u\n", id, row);

		filter_check_graph(priv, path, enabled);
		filter_settings_update(priv, iter, true);
	}
}

static void filter_list_deleted(GtkTreeModel* model, GtkTreePath* path, FilterDialogPrivate *priv)
{
	UNUSED(model);

	if (!gtk_tree_path_get_depth(path))
		return;
	gint row = gtk_tree_path_get_indices(path)[0];
	prdbg("list element deleted, was at row %d", row);

	if (priv->dnd_src == -1) {
		/* Handle filter removal */
		prdbg("Remove filter %d", row);
		const bool enabled = filter_chain_get(priv->chain, row)->enabled;
		filter_chain_remove(priv->chain, row);
		gint n = gtk_tree_model_iter_n_children(model, NULL);
		if (n > row)
			filter_list_fixup(priv, row, n - 1);

		// Not entirely correct to compare path to selected filter?
		filter_check_graph(priv, path, enabled);
	} else {
		// Original ID of deleted row, pre-insert
		gint src_id = priv->dnd_dst <= row ? row - 1 : row;
		// Destination row, after delete
		gint dst_row = priv->dnd_dst <= row ? priv->dnd_dst : priv->dnd_dst - 1;
		/* DND complete */
		if (src_id != priv->dnd_src)  {
			printf("DND in progress but delete isn't DND src row?\n");
			/* handle as non-dnd? Abort dnd? */
			return;
		}
		priv->dnd_src = priv->dnd_dst = -1;
		if (dst_row != src_id) {
			prdbg("DND complete, move row %d to %d", src_id, dst_row);
			filter_list_fixup(priv, src_id, dst_row);
			filter_chain_move(priv->chain, src_id, dst_row);
			filter_check_graph(priv, path, filter_chain_get(priv->chain, dst_row)->enabled);
		} else {
			prdbg("null-DND complete, nothing to do");
		}
	}
}

static void filter_list_inserted(GtkTreeModel* model, GtkTreePath* path, GtkTreeIter* iter, FilterDialogPrivate *priv)
{
	gint id, row = gtk_tree_path_get_indices(path)[0];
	gboolean nonblank;

	gtk_tree_model_get(model, iter, COL_ID, &id, COL_NONBLANK, &nonblank, -1);
	if (nonblank) {
		guint freq;
		gdouble q, gain;
		enum bitype type;
		gboolean enabled;
		gtk_tree_model_get(model, iter,
			COL_TYPE, &type, COL_FREQ, &freq, COL_Q, &q, COL_ENABLED, &enabled, COL_GAIN, &gain, -1);
		prdbg("New filter #%d (row %d): %u %u %f %f", id, row, type, freq, q, gain);
		if (id != row)
			printf("Inserting non-blank element with incorrect ID? %d vs %d\n", id, row);
		if (!filter_defaults[type].hasgain)
			gain = 0;
		filter_chain_set_filter(priv->chain, filter_chain_insert(priv->chain, row), type, freq, q, gain);
		filter_check_graph(priv, path, enabled);
	} else {
		prdbg("New blank element, probably DND, at %d", row);
		priv->dnd_dst = row;
		/* Will get handled when DND completes */
		return;
	}
}

static void fill_filter_liststore(FilterDialogPrivate* priv)
{
	g_signal_handlers_block_matched(G_OBJECT(priv->model), G_SIGNAL_MATCH_DATA, 0,0,NULL,NULL, priv);

	gtk_list_store_clear(GTK_LIST_STORE(priv->model));
	unsigned i = 0;
	const struct biquad_filter *f;
	while ((f = filter_chain_get(priv->chain, i))) {
		prdbg("Adding filter #%u to liststore", i);
		gtk_list_store_insert_with_values(GTK_LIST_STORE(priv->model), NULL, -1,
			COL_TYPE,	f->type,
			COL_FREQ,	f->frequency,
			COL_Q,		f->bw,
			COL_ENABLED,	f->enabled,
			COL_GAIN,	f->gain,
			COL_ID,		i,
			COL_NONBLANK,	TRUE,
			-1);
		i++;
	}
	/* Why add i as data when we already know what the row is?  So we can detect when DND
	 * moves a row by creating a new one and deleting the old one.  */

	g_signal_handlers_unblock_matched(G_OBJECT(priv->model), G_SIGNAL_MATCH_DATA, 0,0,NULL,NULL, priv);
}

/* Check if change to path needs the edit pane to update */
static void filter_edit_check(FilterDialogPrivate *priv, const gchar* path, unsigned index)
{
	if (!priv->selection)
		return;
	g_autoptr(GtkTreePath) tpath = path ? gtk_tree_path_new_from_string(path) : gtk_tree_path_new_from_indices(index, -1);

	if (gtk_tree_path_compare(tpath, priv->selection) == 0) {
		prdbg("need to update edit pane");
		priv->edit_updated = false;
	}
}

static void filter_edited(GtkCellRendererText* renderer, const gchar* path, const gchar* text, FilterDialogPrivate *priv)
{
	GtkTreeIter iter;
	if (!gtk_tree_model_get_iter_from_string(priv->model, &iter, path))
		return;
	guint col = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(renderer), "column"));

	// This is so annoying.  We get the value as text and must convert to the column's type.
	// GValue doesn't do this for us either.
	g_auto(GValue) value = G_VALUE_INIT;
	GType t = gtk_tree_model_get_column_type(priv->model, col);
	g_value_init(&value, t);
	switch (t) {
		case G_TYPE_UINT: g_value_set_uint(&value, strtoul(text, NULL, 0)); break;
		case G_TYPE_DOUBLE: g_value_set_double(&value, strtod(text, NULL)); break;
		default:
			printf("Don't know about this column type\n");
	}

	filter_edit_check(priv, path, -1);
	gtk_list_store_set_value(GTK_LIST_STORE(priv->model), &iter, col, &value);
}

static void filter_toggled(GtkCellRendererToggle* renderer, const gchar* path, FilterDialogPrivate *priv)
{
	GtkTreeIter iter;
	if (!gtk_tree_model_get_iter_from_string(priv->model, &iter, path))
		return;
	guint col = GPOINTER_TO_UINT(g_object_get_data(G_OBJECT(renderer), "column"));
	filter_edit_check(priv, path, -1);

	prdbg("toggled in col %u", col);
	gboolean state;
	gtk_tree_model_get(priv->model, &iter, col, &state, -1);
	gtk_list_store_set(GTK_LIST_STORE(priv->model), &iter, col, !state, -1);
}

static void filter_type_data_func(GtkTreeViewColumn* col, GtkCellRenderer* renderer, GtkTreeModel* model,
				  GtkTreeIter* iter, gpointer column)
{
	UNUSED(col);
	enum bitype type;
	gtk_tree_model_get(model, iter, GPOINTER_TO_UINT(column), &type, -1);
	g_object_set(renderer, "text", filter_names[type], NULL);
}

static void filter_gain_data_func(GtkTreeViewColumn* col, GtkCellRenderer* renderer, GtkTreeModel* model,
				  GtkTreeIter* iter, gpointer column)
{
	UNUSED(col);
	enum bitype type;
	gdouble gain;
	gchar buf[16] = "";

	gtk_tree_model_get(model, iter, GPOINTER_TO_UINT(column), &gain, COL_TYPE, &type, -1);
	gboolean hasgain = filter_defaults[type].hasgain;
	if (hasgain)
		g_snprintf(buf, sizeof(buf), "%.2f", gain);
	g_object_set(renderer, "text", buf, "visible", hasgain, "sensitive", hasgain, NULL);
}

static void filter_selected(GtkTreeSelection* selection, FilterDialogPrivate* priv)
{
	GtkTreeIter iter;
	if (gtk_tree_selection_get_selected(selection, NULL, &iter)) {
		gtk_widget_set_sensitive(priv->editframe, TRUE);
		g_autoptr(GtkTreePath) path = gtk_tree_model_get_path(priv->model, &iter);
		g_autofree gchar *old = NULL, *new = NULL;
		prdbg("selection was %s now %s", !priv->selection ? "none" : (old = gtk_tree_path_to_string(priv->selection)),
						new = gtk_tree_path_to_string(path));
		if (priv->selection && gtk_tree_path_compare(path, priv->selection) == 0) {
			prdbg("selection not changed");
			return;
		}
		gtk_tree_path_free(priv->selection);
		priv->selection = g_steal_pointer(&path);
		gboolean enabled;
		gtk_tree_model_get(priv->model, &iter, COL_ENABLED, &enabled, -1);
		priv->edit_updated = false;
		if (!priv->fullchain)
			priv->graph_updated = false;
		filter_settings_update(priv, &iter, false);
	} else {
		prdbg("nothing selected");
		gtk_tree_path_free(priv->selection);
		priv->selection = NULL;
		gtk_widget_set_sensitive(priv->editframe, FALSE);
		if (!priv->fullchain) {
			gtk_image_clear(priv->image);
			priv->graph_updated = true;
		}
		priv->edit_updated = false;
	}

}

static GtkTreeView *create_listview(FilterDialogPrivate *priv)
{
	GtkTreeView *view = GTK_TREE_VIEW(gtk_tree_view_new_with_model(priv->model));
	gtk_tree_view_set_reorderable(view, true);
	gtk_tree_view_set_grid_lines(view, GTK_TREE_VIEW_GRID_LINES_VERTICAL);
	GtkTreeSelection *selection = gtk_tree_view_get_selection(view);
	gtk_tree_selection_set_mode(selection, GTK_SELECTION_SINGLE);
	g_signal_connect(selection, "changed", G_CALLBACK(filter_selected), priv);

	GtkCellRenderer *renderer;
	int col;
	renderer = gtk_cell_renderer_text_new();
	col = gtk_tree_view_insert_column_with_data_func(view, -1, "Filter Type", renderer,
		filter_type_data_func, GUINT_TO_POINTER(COL_TYPE), NULL);
	gtk_tree_view_column_set_expand(gtk_tree_view_get_column(view, col - 1), TRUE);
	gtk_tree_view_column_set_alignment(gtk_tree_view_get_column(view, col - 1), 0.5f);

	renderer = gtk_cell_renderer_spin_new();
	g_object_set_data(G_OBJECT(renderer), "column", GUINT_TO_POINTER(COL_FREQ));
	g_signal_connect(renderer, "edited", G_CALLBACK(filter_edited), priv);
	g_object_set(renderer,
		"editable", true,
		"adjustment", gtk_adjustment_new(0, 0, 24000, 10, 1000, 0),
		"digits", 0,
		"max-width-chars", 6,
		"xalign", 1.0f,
		NULL);
	col = gtk_tree_view_insert_column_with_attributes(view, -1, "Frequency", renderer,
		"text", COL_FREQ,
		NULL);
	gtk_tree_view_column_set_expand(gtk_tree_view_get_column(view, col - 1), TRUE);
	gtk_tree_view_column_set_alignment(gtk_tree_view_get_column(view, col - 1), 0.5f);

	renderer = gtk_cell_renderer_spin_new();
	g_object_set_data(G_OBJECT(renderer), "column", GUINT_TO_POINTER(COL_Q));
	g_signal_connect(renderer, "edited", G_CALLBACK(filter_edited), priv);
	g_object_set(renderer,
		"editable", true,
		"adjustment", gtk_adjustment_new(.707, 0, 1, .001, .01, 0),
		"max-width-chars", 9,
		"digits", 5,
		"xalign", 1.0f,
		NULL);
	col = gtk_tree_view_insert_column_with_attributes(view, -1, "Q/BW", renderer,
		"text", COL_Q,
		NULL);
	gtk_tree_view_column_set_expand(gtk_tree_view_get_column(view, col - 1), TRUE);
	gtk_tree_view_column_set_alignment(gtk_tree_view_get_column(view, col - 1), 0.5f);

	renderer = gtk_cell_renderer_spin_new();
	g_object_set_data(G_OBJECT(renderer), "column", GUINT_TO_POINTER(COL_GAIN));
	g_signal_connect(renderer, "edited", G_CALLBACK(filter_edited), priv);
	g_object_set(renderer,
		"editable", true,
		"adjustment", gtk_adjustment_new(0, -35, 35, 1, .1, 0),
		"max-width-chars", 9,
		"digits", 2,
		"xalign", 1.0f,
		NULL);
	col = gtk_tree_view_insert_column_with_data_func(view, -1, "Gain", renderer,
		filter_gain_data_func, GUINT_TO_POINTER(COL_GAIN), NULL);
	gtk_tree_view_column_set_expand(gtk_tree_view_get_column(view, col - 1), TRUE);
	gtk_tree_view_column_set_alignment(gtk_tree_view_get_column(view, col - 1), 0.5f);

	renderer = gtk_cell_renderer_toggle_new();
	g_object_set_data(G_OBJECT(renderer), "column", GUINT_TO_POINTER(COL_ENABLED));
	g_signal_connect(renderer, "toggled", G_CALLBACK(filter_toggled), priv);
	g_object_set(renderer, "activatable", true, NULL);
	gtk_tree_view_insert_column_with_attributes(view, -1, "Enabled", renderer,
		"active", COL_ENABLED,
		NULL);
#if 0
	renderer = gtk_cell_renderer_text_new();
	gtk_tree_view_insert_column_with_attributes(view, -1, "ID", renderer,
		"text", COL_ID,
		NULL);
#endif

	return view;
}

void filter_dialog_set_chain(FilterDialog* filter_dialog, struct filter_chain* chain)
{
	FilterDialogPrivate* priv = filter_dialog_get_instance_private(FILTER_DIALOG(filter_dialog));
	priv->chain = chain;
	fill_filter_liststore(priv);
	// FIXME: update graphs? unselect filters?
}
