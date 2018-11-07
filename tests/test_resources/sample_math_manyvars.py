import numpy


def process_row(input_numpy_array):
    temporary_variable_array = numpy.zeros(2)
    return_numpy_array = numpy.zeros(3)
    five = 10. / 2.
    temporary_variable_array[0] = 3. + 3.
    temporary_variable_array[1] = input_numpy_array[0] + input_numpy_array[1]
    return_numpy_array[0] = input_numpy_array[0] + 6.
    return_numpy_array[1] = temporary_variable_array[0] - five
    return_numpy_array[2] = temporary_variable_array[1] * input_numpy_array[1]
    return return_numpy_array


if __name__ == '__main__':
    sample_row = numpy.array([1., 2., 3.], dtype=numpy.float64)
    process_row(sample_row)
