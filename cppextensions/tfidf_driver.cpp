#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h> // For explicitly sized integer types.
#include <stdlib.h> // For malloc
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <string.h>

using namespace std;

// Defines Weld's primitive numeric types.
typedef bool        i1;
typedef uint8_t     u8;
typedef int8_t      i8;
typedef uint16_t   u16;
typedef int16_t    i16;
typedef uint32_t   u32;
typedef int32_t    i32;
typedef uint64_t   u64;
typedef int64_t    i64;
typedef float      f32;
typedef double     f64;

// Defines the basic Vector struture using C++ templates.
template<typename T>
struct vec {
  T *ptr;
  i64 size;
};
struct struct0 {
  vec<vec<i8>> _0;
};
typedef struct0 input_type;


struct WeldInputArgs {
	void* input;
	i32 nworkers;
	i64 memlimit;
	void* run_id;
};

struct WeldOutputArgs {
	void* output;
	i64 run;
	i64 errno;
};

extern "C" struct WeldOutputArgs* WELD_ENTRY_POINT(struct WeldInputArgs*);
extern "C" void* weld_runst_init(i32, i64);

static PyObject *
caller_func(PyObject *self, PyObject* args)
{
    PyObject* vocab_list = NULL;
    PyObject *idf = NULL;
    PyArrayObject *idf_array = NULL;
    if (!PyArg_ParseTuple(args, "OO!", &vocab_list, &PyArray_Type, &idf)) {
        return NULL;
    }
    if((idf_array = (PyArrayObject*) PyArray_FROM_OTF(idf, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        return NULL;
    }
    Py_ssize_t vocab_list_length = PyList_Size(vocab_list);
    input_type weld_input;
    weld_input._0.size = vocab_list_length;
    vec<i8>* weld_strings_list = (vec<i8>*) malloc(vocab_list_length * sizeof(vec<i8>));
    weld_input._0.ptr = weld_strings_list;
    for(Py_ssize_t i = 0; i < vocab_list_length; i++) {
        PyObject* python_string = PyList_GetItem(vocab_list, i);
        Py_ssize_t python_string_len;
        char* python_string_data = PyUnicode_AsUTF8AndSize(python_string, &python_string_len);
        weld_strings_list[i].size = python_string_len;
        weld_strings_list[i].ptr = (i8*) malloc(sizeof(char) * python_string_len);
        strncpy((char*) weld_strings_list[i].ptr, python_string_data, python_string_len);
    }

    struct WeldInputArgs weld_input_args;
    weld_input_args.input = &weld_input;
    weld_input_args.nworkers = 1;
    weld_input_args.memlimit = 100000000;
    weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);

    WeldOutputArgs* weld_output_args = WELD_ENTRY_POINT(&weld_input_args);

    f64* idf_data = (f64*) PyArray_DATA(idf_array);
    int idf_len = PyArray_DIMS(idf_array)[0];
    f64* copied_idf_data = (f64*) malloc(idf_len * sizeof(double));
    memcpy(copied_idf_data, idf_data, idf_len * sizeof(double));

    vec<f64>* weld_idf_vec = (vec<f64>*) malloc(sizeof(vec<f64>));
    weld_idf_vec->size = idf_len;
    weld_idf_vec->ptr = copied_idf_data;
    // Weld returns a pointer to a pointer to the dict.
    PyObject* return_tuple = PyTuple_New(2);
    PyTuple_SetItem(return_tuple, 0, PyLong_FromVoidPtr(* (void**) weld_output_args->output));
    PyTuple_SetItem(return_tuple, 1, PyLong_FromVoidPtr((void*) weld_idf_vec));
    return return_tuple;
}
static PyMethodDef CallerMethods[] = {
 { "caller_func", caller_func, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef tfidf_driver_module = {
    PyModuleDef_HEAD_INIT,
    "tfidf_driver",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_tfidf_driver(void)
{
    import_array();
    return PyModule_Create(&tfidf_driver_module);
}
