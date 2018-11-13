import numpy
from timeit import default_timer as timer


def process_row(input_numpy_array):
    return_numpy_array = numpy.zeros(2)
    return_numpy_array[0] = input_numpy_array[0] + input_numpy_array[1]
    return_numpy_array[1] = input_numpy_array[1] * input_numpy_array[2]
    return input_numpy_array


sample_row = numpy.ones(100, dtype=numpy.float64)
NUMITER = 10000
timesum = 0
start = timer()
for i in range(NUMITER):
    little_start = timer()
    process_row(sample_row)
    little_end = timer()
    diff = little_end - little_start
    # print(diff)
    if i > 1000:
        timesum += diff
end = timer()
print(timesum / (NUMITER - 1000))
