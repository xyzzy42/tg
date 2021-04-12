//SPDX-License-Identifier: GPL-2.0-or-later
/*
    Python graphics for C Gtk applications
    Copyright (C) 2021  Trent Piepho <tpiepho@gmail.com>

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation; either version 2 of the License, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details.

    You should have received a copy of the GNU General Public License along with
    this program; if not, write to the Free Software Foundation, Inc., 51
    Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "tg.h"

static bool python_initialized;

#if PY_VERSION_HEX < 0x03090000
static inline PyObject* PyObject_CallNoArgs(PyObject* func)
{ return PyObject_CallObject(func, NULL); }
#endif

// Python module name for the Python extension C functions
#define EXTMODULE	"tg"

// State data attached to the extension module
typedef const struct main_window* ModState;	// just one field, so not a struct

// Get main_window pointer from the tg module object, which is the self pointer for any
// call to a method in the module
const struct main_window* get_w(PyObject* self)
{
	ModState* state = PyModule_GetState(self);
	return *state;
}

// Converter function for PyArg_ParseTuple("O&").  Numeric type to a size_t.  Lets one
// accept both np.uint64() and int().
static int ConvertSizeT(PyObject* object, void* address)
{
	const uint64_t value = PyLong_AsSize_t(PyNumber_Long(object));
	if (value == (size_t)-1 && PyErr_Occurred()) {
		printf("couldn't convert to size_t\n");
		return 0;
	}
	*(size_t*)address = value;
	return 1;
}

/******************************************************************************
 * Functions for python extension module that python code can call to interact
 * with the C app.  */

/* Return a matplotlib colorspec for a GtkEntry's normal foreground color.  I
 * don't like how this creates a dummy GtkEntry but don't know the alternative
 * to gtk_widget_get_style_context() to get the GtkStyleContext for a widget by
 * type.  */
static PyObject* get_fgcolor(PyObject* self, PyObject* noargs)
{
	UNUSED(self); UNUSED(noargs);
	static GdkRGBA color;
	static bool init = false;

	if (!init) {
	    GtkWidget* dummy = gtk_entry_new();
	    GtkStyleContext* style = gtk_widget_get_style_context(dummy);
	    gtk_style_context_get_color(style, GTK_STATE_FLAG_NORMAL, &color);
	    g_object_ref_sink(dummy);
	    init = true;
	}

	return Py_BuildValue("dddd", color.red, color.green, color.blue, color.alpha);
}

/* Returns audio as a np.array(dtype=np.float32).  Returns None if the audio isn't in
 * the ring buffer.  The object owns the array data and will free it when the array is
 * GCed.  numpy didn't document how the data is freed exactly, so I hope it's with
 * free().  */
static PyObject* pack_audio(uint64_t timestamp, int length)
{
	float *data = get_audio_data(timestamp, length);
	if (!data)
		Py_RETURN_NONE;
	npy_intp dims[1] = { length };
	PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);
	PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
	return array;
}

// getaudio(timestamp: int, length: int): np.array(float32) or None
// Get audio at a timestamp.
// Maybe extend this to allow negative timestamp to mean from the end?
static PyObject* get_audio(PyObject* self, PyObject* args)
{
	UNUSED(self);
	unsigned long long timestamp;
	int length;

	if (!PyArg_ParseTuple(args, "O&i", ConvertSizeT, &timestamp, &length)) {
		PyErr_Print();
		return NULL;
	}
	return pack_audio(timestamp, length);
}

// getlastaudio(length: int): np.array(float32), timestamp or None
// Get most recent length samples of audio.  Returns timestamp of first (oldest) sample.
static PyObject* get_lastaudio(PyObject* self, PyObject* arg)
{
	UNUSED(self);
	const long length = PyLong_AsLong(PyNumber_Long(arg));
	if (length == -1 && PyErr_Occurred())
		return NULL;

	uint64_t timestamp;
	float* data = get_last_audio_data(length, &timestamp);
	if (!data)
		Py_RETURN_NONE;
	npy_intp dims[1] = { length };
	PyObject* array = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, data);
	PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);
	return Py_BuildValue("NK", array, (unsigned long long)timestamp);
}

// getsamplerate(): float
// Return current sample rate.	This is in the units timestamps use and reflects the
// lite mode decimation (I think) and calibration.
static PyObject* get_samplerate(PyObject* self, PyObject* noargs)
{
	UNUSED(noargs);
	const struct main_window* w = get_w(self);
	return PyFloat_FromDouble(w->active_snapshot->sample_rate);
}

// getevents(): np.array(dtype=np.uint64) or None
// Timestamps of every beat.
// Data is borrowed and object shouldn't be saved across calls without making a copy.
static PyObject* get_events(PyObject* self, PyObject* args)
{
	UNUSED(args);
	const struct main_window* w = get_w(self);
	const struct snapshot* snst = w->active_snapshot;
	if (snst->events_count == 0)
		Py_RETURN_NONE;
	// FIXME: handle wrap of circular buffer
	// FIXME: might not start at 0, should end at events_wp and go backward
	if (snst->events_wp != snst->events_count - 1)
		return NULL; // lazy, see above
	// lifetime of data might not be long enough, need care
	npy_intp dims[1] = { snst->events_count };
	return PyArray_SimpleNewFromData(1, dims, NPY_UINT64, snst->events);
}

// getlastevents(): tuple(np.uint64, np_uint64)
// Timestamps of the last, as in oldest, tic and toc events still in the processing
// buffer.
static PyObject* get_last_events(PyObject* self, PyObject* noargs)
{
	UNUSED(noargs);
	const struct main_window* w = get_w(self);
	const struct snapshot* snst = w->active_snapshot;
	return Py_BuildValue("KK", 
		(unsigned long long)snst->pb->last_tic, (unsigned long long)snst->pb->last_toc);
}

// getbeataudio(timestamp) : (np.array(dtype=float32), int)
// Return what tg considers the beat audio for the one beat at the supplied timestamp.
// The 2nd item is the offset from the start that corresponds to timestamp.
static PyObject* get_beat_audio(PyObject* self, PyObject* arg)
{
	const struct main_window* w = get_w(self);
	const struct snapshot* snst = w->active_snapshot;
	const uint64_t timestamp = PyLong_AsSize_t(PyNumber_Long(arg));

	if (timestamp == (size_t)-1 && PyErr_Occurred())
		return NULL;

	const unsigned offset = NEGATIVE_SPAN * snst->sample_rate / 1000;
	const unsigned ticklen = (NEGATIVE_SPAN + POSITIVE_SPAN) * snst->sample_rate / 1000;
	if (timestamp < offset)
		Py_RETURN_NONE; // throw range exception or something
	return Py_BuildValue("Ni", pack_audio(timestamp - offset, ticklen), offset);
}

// getpulses(): (float, float)
// Return tic and toc pulse lengths, in seconds
static PyObject* get_pulses(PyObject* self, PyObject* noargs)
{
	UNUSED(noargs);
	const struct main_window* w = get_w(self);
	const struct snapshot* snst = w->active_snapshot;

	return Py_BuildValue("dd", snst->pb->tic_pulse, snst->pb->toc_pulse);
}

static PyMethodDef methods[] = {
	{"fgcolor",	  get_fgcolor,	  METH_NOARGS, "Get GtkImage's normal state foreground color" },
	{"getaudio",	  get_audio,	  METH_VARARGS,"Return nparray of audio data at a timestamp" },
	{"getlastaudio",  get_lastaudio,  METH_O,      "Return most recent audio data and timestamp" },
	{"getevents",	  get_events,	  METH_NOARGS, "Get nparray of current beat timestamps" },
	{"getlastevents", get_last_events,METH_NOARGS, "Return last tic and toc times" },
	{"getsr",	  get_samplerate, METH_NOARGS, "Get calibrated sample rate" },
	{"getbeataudio",  get_beat_audio, METH_O,      "Get audio for a tick, supply timestamp" },
	{"getpulses",	  get_pulses,	  METH_NOARGS, "Get length of samples of tic,toc pulses" },
	{ }
};

static PyModuleDef tg_moduledef = {
	.m_base = PyModuleDef_HEAD_INIT,
	.m_name = EXTMODULE,
	.m_doc = NULL,
	.m_size = sizeof(ModState),
	.m_methods = methods,
};

// Create a extension module from C code, and initialize its state as a copy of
// init_state
static PyObject* create_module(PyModuleDef* def, const ModState init_state)
{
	PyObject* module = PyModule_Create(def);
	ModState* state = PyModule_GetState(module);
	*state = init_state;
	return module;
}

/** Code for loading built in python modules.
 *
 * MODULE(name, method1, method2, ...);
 *
 * Will define a type and a static struct, named "name", that contains fields with names
 * "method1", etc.  that are python objects for those methods as found in the module. 
 *
 * Use with LOADMODULE(name), which loads the module code from a global named "name_py"
 * and sets all the pointers.
 */

// For counting the number of arguments in a __VA_ARGS__ list
#define GET_MACRO(_,_1,_2,_3,_4,_5,NAME,...) NAME
#define COUNT_ARGS(...) \
  GET_MACRO(_,##__VA_ARGS__,5,4,3,2,1,0)
// Make a FOREACH macro
#define FE_0(WHAT)
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
//... repeat as needed
#define FOR_EACH(action,...) \
  GET_MACRO(_,##__VA_ARGS__,FE_5,FE_4,FE_3,FE_2,FE_1,FE_0)(action,__VA_ARGS__)

// Declare an anonymous union containing an array and a struct.  The struct has a field for each
// variable argument, created by the macro DECL.  The array has the specified type and name and
// a length equal to the number of arguments.
#define UNION_ARRAY(array, DECL, ...) \
	union { \
		array[COUNT_ARGS(__VA_ARGS__)]; \
		struct { \
			FOR_EACH(DECL, ## __VA_ARGS__) \
		}; \
	}

#define OBJDECL(obname)	PyObject* obname;
#define IDDECL(idname)	const char* idname ## _id;
#define IDINIT(idname)	.idname ## _id = #idname,
#define MODULE(name, ...) \
extern const char name ## _py[]; \
static struct name ## _t { \
	const char* code; \
	const char* name; \
	UNION_ARRAY(const char* ids, IDDECL, __VA_ARGS__); \
	UNION_ARRAY(PyObject* objs, OBJDECL, __VA_ARGS__); \
} name = { \
	.code = name ## _py, \
	.name = #name, \
	FOR_EACH(IDINIT, __VA_ARGS__) \
}

// Number of methods in module
#define MODULE_COUNT(module)	ARRAY_SIZE(module.ids)

struct emptymod {
	const char* code;
	const char* name;
	const char* ids[];
};

// Create a module from supplied python code
static PyObject* import_module(const char* code, const char* name)
{
	PyObject *bytecode = Py_CompileString(code, name, Py_file_input);
	if (!bytecode) {
		printf("Error compiling Python code from %s.py:\n", name);
		goto error;
	}
	PyObject *module = PyImport_ExecCodeModule(name, bytecode);
	Py_DECREF(bytecode);
	if (!module) {
		printf("Error running %s.py:\n", name);
		goto error;
	}

	return module;

error:
	return NULL;
}

/* Get a function from a python module */
static PyObject* get_method(PyObject* module, const char* name)
{
	PyObject *func = PyObject_GetAttrString(module, name);
	if (!func || !PyCallable_Check(func)) {
		printf("Unable to get function '%s' from Python module\n", name);
		return NULL;
	}
	return func;
}

static bool loadmodule(struct emptymod *module, size_t n) {
	PyObject* pymodule = import_module(module->code, module->name);
	if (!pymodule)
		return false;

	bool fail = false;
	PyObject** objs = (PyObject**)(module->ids + n);
	for(unsigned i=0; i < n; i++) {
		objs[i] = get_method(pymodule, module->ids[i]);
		if (!objs[i]) {
			printf("Failed to get method '%s' from module '%s'\n", module->ids[i], module->name);
			fail = true;
		}
	}
	if (fail)
		for(unsigned i=0; i < n; i++) Py_XDECREF(objs[i]);
	Py_DECREF(pymodule);

	return !fail;
}
#define LOADMODULE(module)	loadmodule((struct emptymod*)&module, MODULE_COUNT(module))

static void unloadmodule(struct emptymod *module, size_t n) {
	PyObject** objs = (PyObject**)(module->ids + n);
	for(unsigned i=0; i < n; i++) 
		Py_XDECREF(objs[i]);
}
#define UNLOADMODULE(module)	unloadmodule((struct emptymod*)&module, MODULE_COUNT(module))

// Yuck, doesn't seem to be any way to pass something to this callback
// other than a global.
static ModState module_w_ptr;
// Create function for inittab
static PyObject* create_tgmodule(void)
{
	return create_module(&tg_moduledef, module_w_ptr);
}

bool python_init(const struct main_window* w)
{
	if (python_initialized)
		return true;

	module_w_ptr = w;  // Argument for create_tgmodule()
	PyImport_AppendInittab(EXTMODULE, create_tgmodule);
	Py_Initialize();
	import_array();  // Initialize numpy

	python_initialized = true;
	return true;
error:
	PyErr_Print();
	return false;
}

void python_finish(void)
{
	Py_FinalizeEx();
}

// A gdk pixmap deleter function that will delete a PyObject
static void del_pyobject(guchar *pixels, gpointer data)
{
	UNUSED(pixels);
	Py_XDECREF((PyObject*)data);
}

// Return a pixbuf created from the image in the memview object.
// The pixbuf steals the reference to the memview!
// Needs to be 8-bit RGBA and the memview needs to describe the image dimensions.  The
// pixbuf will deref the memview when it's freed.
static GdkPixbuf* create_pixbuf_from_memview(PyObject* memview)
{
	if (!PyMemoryView_Check(memview))
		return NULL; // Create exception?
	Py_buffer *mem = PyMemoryView_GET_BUFFER(memview);
	if (mem->ndim != 3 || mem->shape[2] != 4)
		return NULL;
	return gdk_pixbuf_new_from_data(mem->buf,
		GDK_COLORSPACE_RGB,
		true,
		8,
		mem->shape[1],
		mem->shape[0],
		mem->strides[0],
		del_pyobject, memview);
}

// Set a GtkImage to an image stored in a Python memview.  This steals the reference to the
// memview and gives it to the GtkImage.  Prints error on a NULL memview.
static void image_set_from_memview(GtkImage* image, PyObject* memview)
{
	if (!memview) {
		PyErr_Print();
		return;
	}
	GdkPixbuf* pixbuf = create_pixbuf_from_memview(memview);
	if (pixbuf) {
		gtk_image_set_from_pixbuf(image, pixbuf);
		g_object_unref(pixbuf);
	}
}
