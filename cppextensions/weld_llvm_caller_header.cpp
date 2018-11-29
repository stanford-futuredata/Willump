#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
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

struct struct0 {
  vec<WELD_INPUT_TYPE_0> _0;
  void* _1;
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
typedef vec<WELD_OUTPUT_TYPE> return_type;
