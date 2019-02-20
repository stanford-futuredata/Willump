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

// Defines the basic Vector structure using C++ templates.
template<typename T>
struct vec {
  T *ptr;
  i64 size;
};
struct struct0 {
  INPUT_STRUCT_CONTENTS
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
    INPUT_PARSING_CONTENTS

    struct WeldInputArgs weld_input_args;
    weld_input_args.input = &weld_input;
    weld_input_args.nworkers = 1;
    weld_input_args.memlimit = 10000000000;
    weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);

    WeldOutputArgs* weld_output_args = WELD_ENTRY_POINT(&weld_input_args);
    // Weld returns a pointer to a pointer to the dict.
    return PyLong_FromVoidPtr(* (void**) weld_output_args->output);
}
static PyMethodDef CallerMethods[] = {
 { "caller_func", caller_func, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef hash_join_dataframe_indexer_module = {
    PyModuleDef_HEAD_INIT,
    "hash_join_dataframe_indexer",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_hash_join_dataframe_indexer(void)
{
    import_array();
    return PyModule_Create(&hash_join_dataframe_indexer_module);
}
