import numpy


def process_row(input_numpy_array):
    return_numpy_array = numpy.zeros(3)
    return_numpy_array[0] = 1. + 1.
    return_numpy_array[1] = input_numpy_array[0] - 5.
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


if __name__ == '__main__':
    sample_row = numpy.array([1., 2., 3.], dtype=numpy.float64)
    process_row(sample_row)
