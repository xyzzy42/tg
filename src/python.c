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

static PyMethodDef methods[] = {
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
