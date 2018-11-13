#include <Python.h>
#include <stdint.h> // For explicitly sized integer types.
#include <stdlib.h> // For malloc
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <iostream>

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

template<typename T>
vec<T> new_vec(i64 size) {
  vec<T> t;
  t.ptr = (T *)malloc(size * sizeof(T));
  t.size = size;
  return t;
}

vec<f64> new_vecf(i64 size) {
  vec<f64> t;
  t.ptr = (f64 *)malloc(size * sizeof(f64));
  t.size = size;
  return t;
}

struct struct0 {
  vec<f64> _0;
};

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

extern "C" struct WeldOutputArgs* run(struct WeldInputArgs*);
extern "C" void* weld_runst_init(i32, i64);

// Aliases for argument and return types.
typedef struct0 input_type;
typedef vec<f64> return_type;

static PyObject *
weld_llvm_caller(PyObject *self, PyObject* args)
{
    printf("Hello world\n");
    return PyUnicode_FromString("hello world");
}

static PyMethodDef CallerMethods[] = {
 { "weld_llvm_caller", weld_llvm_caller, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef weld_llvm_caller_module = {
    PyModuleDef_HEAD_INIT,
    "weld_llvm_caller",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_weld_llvm_caller(void)
{
    return PyModule_Create(&weld_llvm_caller_module);
}
