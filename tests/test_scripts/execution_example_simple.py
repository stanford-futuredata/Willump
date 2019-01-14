import numpy
from timeit import default_timer as timer
import willump.evaluation.willump_executor


@willump.evaluation.willump_executor.willump_execute()
def process_row(input_numpy_array):
    return_numpy_array = numpy.zeros(2)
    return_numpy_array[0] = input_numpy_array[0] + input_numpy_array[1]
    return_numpy_array[1] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array

def main():
    sample_row = numpy.ones(101, dtype=numpy.float64)
    NUMITER = 10000
    sample_row[10] = 5
    sample_row[100] = 6
    sample_row[11] = 8
    # Force compilation.
    process_row(sample_row)
    process_row(sample_row)
    start = timer()
    for _ in range(NUMITER):
        process_row(sample_row)
    end = timer()
    print((end - start) / NUMITER)


if __name__ == '__main__':
    main()

