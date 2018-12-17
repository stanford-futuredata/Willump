import numpy
import math
import willump.evaluation.willump_executor


@willump.evaluation.willump_executor.willump_execute
def process_row(input_numpy_array):
    return_numpy_array = numpy.zeros(3, dtype=numpy.float64)
    return_numpy_array[0] = 1. + 1.
    return_numpy_array[1] = input_numpy_array[0] - math.sqrt(25) + 5 - 5.0
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


def main():
    with open("temp_execution_correctness_file.tmp", "w") as outfile:
        sample_row = numpy.array([1, 2, 3], dtype=numpy.int32)
        print(process_row(sample_row), file=outfile)
        sample_row2 = numpy.array([4, 5, 6], dtype=numpy.int32)
        print(process_row(sample_row2), file=outfile)


if __name__ == '__main__':
    main()
