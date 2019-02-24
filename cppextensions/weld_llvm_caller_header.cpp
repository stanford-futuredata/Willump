#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h> // For explicitly sized integer types.
#include <stdlib.h> // For malloc
#include <stdio.h>
#include <ctime>
#include <chrono>
#include <iostream>
#include <atomic>

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

extern "C" void* weld_runst_init(i32, i64);

i64* csr_matrix_row_maker(i64* indptr, int new_row_len, int indptr_len) {
    i64* new_row = (i64*) malloc(new_row_len * sizeof(i64));
    int indptr_index = 0;
    for(int i = 0; i < new_row_len; i++) {
        if (indptr_index < indptr_len and i >= indptr[indptr_index + 1])
            indptr_index += 1;
        new_row[i] = indptr_index;
    }
    return new_row;
}
