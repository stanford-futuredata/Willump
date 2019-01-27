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

static PyObject *
caller_func(PyObject *self, PyObject* args)
{
    PyObject *weights = NULL;
    PyObject *intercept = NULL;
    PyArrayObject *weights_array = NULL;
    PyArrayObject *intercept_array = NULL;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &weights, &PyArray_Type, &intercept)) {
        return NULL;
    }
    if((weights_array = (PyArrayObject*) PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        return NULL;
    }
    if((intercept_array = (PyArrayObject*) PyArray_FROM_OTF(intercept, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)) == NULL) {
        return NULL;
    }
    f64* weights_data = (f64*) PyArray_DATA(weights_array);
    int weights_len = PyArray_DIMS(weights_array)[0];
    f64* copied_weights_data = (f64*) malloc(weights_len * sizeof(double));
    memcpy(copied_weights_data, weights_data, weights_len * sizeof(double));

    vec<f64>* weld_weights_vec = (vec<f64>*) malloc(sizeof(vec<f64>));
    weld_weights_vec->size = weights_len;
    weld_weights_vec->ptr = copied_weights_data;

    f64* intercept_data = (f64*) PyArray_DATA(intercept_array);
    int intercept_len = PyArray_DIMS(intercept_array)[0];
    f64* copied_intercept_data = (f64*) malloc(intercept_len * sizeof(double));
    memcpy(copied_intercept_data, intercept_data, intercept_len * sizeof(double));

    vec<f64>* weld_intercept_vec = (vec<f64>*) malloc(sizeof(vec<f64>));
    weld_intercept_vec->size = intercept_len;
    weld_intercept_vec->ptr = copied_intercept_data;


    PyObject* return_tuple = PyTuple_New(2);
    PyTuple_SetItem(return_tuple, 0, PyLong_FromVoidPtr((void*) weld_weights_vec));
    PyTuple_SetItem(return_tuple, 1, PyLong_FromVoidPtr((void*) weld_intercept_vec));
    return return_tuple;
}
static PyMethodDef CallerMethods[] = {
 { "caller_func", caller_func, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef encode_logistic_regression_model_module = {
    PyModuleDef_HEAD_INIT,
    "encode_logistic_regression_model",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_encode_logistic_regression_model(void)
{
    import_array();
    return PyModule_Create(&encode_logistic_regression_model_module);
}
