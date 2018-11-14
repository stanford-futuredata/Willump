import numpy
from timeit import default_timer as timer


def process_row(input_numpy_array):
    return_numpy_array = numpy.zeros(2)
    return_numpy_array[0] = input_numpy_array[0] + input_numpy_array[1]
    return_numpy_array[1] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


sample_row = numpy.ones(101, dtype=numpy.float64)
NUMITER = 10000
timesum = 0
sample_row[10] = 5
sample_row[100] = 6
sample_row[11] = 8
# Force a Numba precompile (when comparing against Numba).
process_row(sample_row)
start = timer()
for _ in range(NUMITER):
    process_row(sample_row)
end = timer()
print((end - start) / NUMITER)
