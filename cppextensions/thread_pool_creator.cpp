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

typedef struct weld_thread_runner {
    bool done;
    struct WeldOutputArgs* (*run_function)(struct WeldInputArgs*);
    struct WeldInputArgs* argument;
    struct WeldOutputArgs* output;
    bool ready;
} weld_thread_runner;

static void* thread_start(void* arg) {
    weld_thread_runner* thread_runner = (weld_thread_runner*) arg;
    while(1) {
        if (thread_runner->ready) {

            printf("RUNNER %d\n", sched_getcpu());
            thread_runner->ready = false;
            thread_runner->output = thread_runner->run_function(thread_runner->argument);
            thread_runner->done = true;
        }
    }
}

static PyObject *
caller_func(PyObject *self, PyObject* args)
{
    int num_threads;
    if (!PyArg_ParseTuple(args, "i", &num_threads)) {
        return NULL;
    }

    weld_thread_runner* thread_runner_array = (weld_thread_runner*) calloc(num_threads, sizeof(weld_thread_runner));

    pthread_attr_t attr;
    pthread_attr_init(&attr);

    pthread_t* pthread_ids = (pthread_t*) calloc(num_threads, sizeof(pthread_t));

    for(int t_num = 0; t_num < num_threads; t_num++) {
        pthread_create(&pthread_ids[t_num], &attr, &thread_start, &thread_runner_array[t_num]);
    }
    return PyLong_FromVoidPtr(thread_runner_array);
}
static PyMethodDef CallerMethods[] = {
 { "caller_func", caller_func, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef thread_pool_creator_module = {
    PyModuleDef_HEAD_INIT,
    "thread_pool_creator",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_thread_pool_creator(void)
{
    import_array();
    return PyModule_Create(&thread_pool_creator_module);
}
