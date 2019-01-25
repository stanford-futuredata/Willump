import unittest
import numpy
import math

import willump.evaluation.willump_executor as wexec


@wexec.willump_execute()
def sample_math_basic(input_numpy_array):
    return_numpy_array = numpy.zeros(3)
    return_numpy_array[0] = 1. + 1.
    return_numpy_array[1] = input_numpy_array[0] - 5.
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


@wexec.willump_execute()
def sample_math_manyvars(input_numpy_array):
    temporary_variable_array = numpy.zeros(2)
    return_numpy_array = numpy.zeros(3)
    five = 10. / 2. * math.log(2.718281828459045)
    temporary_variable_array[0] = 3. + math.sqrt(9.)
    temporary_variable_array[1] = input_numpy_array[0] + input_numpy_array[1]
    return_numpy_array[0] = input_numpy_array[0] + 6.
    return_numpy_array[1] = temporary_variable_array[0] - five + five - five
    return_numpy_array[2] = temporary_variable_array[1] * input_numpy_array[1]
    return return_numpy_array


@wexec.willump_execute()
def sample_math_ints(input_numpy_array):
    return_numpy_array = numpy.zeros(3, dtype=numpy.int32)
    return_numpy_array[0] = input_numpy_array[0] + input_numpy_array[0]
    return_numpy_array[1] = input_numpy_array[2] - input_numpy_array[1]
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


@wexec.willump_execute()
def sample_math_mixed(input_numpy_array):
    intermediate_numpy_array = numpy.zeros(1, dtype=numpy.float64)
    intermediate_numpy_array[0] = 5 + 5
    three_float = 2 + 1.0
    return_numpy_array = numpy.zeros(3, dtype=numpy.float64)
    return_numpy_array[0] = input_numpy_array[0] + 3.1
    return_numpy_array[1] = input_numpy_array[2] - three_float + math.sqrt(4)
    return_numpy_array[2] = input_numpy_array[1] * intermediate_numpy_array[0]
    return return_numpy_array


@wexec.willump_execute()
def sample_string_split(input_string):
    output_strings = input_string.split()
    return output_strings


@wexec.willump_execute()
def sample_string_lower(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].lower()
    return output_strings


@wexec.willump_execute()
def sample_string_remove_char(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].replace(".", "")
    return output_strings


@wexec.willump_execute()
def sample_scalar_append(input_array, input_value):
    output_array = numpy.append(input_array, input_value)
    return output_array


class GraphInferenceTests(unittest.TestCase):
    def test_basic_math_inference(self):
        print("\ntest_basic_math_inference")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        sample_math_basic(basic_vec)
        sample_math_basic(basic_vec)
        weld_output = sample_math_basic(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., -4., 6.]))

    def test_manyvars_math_inference(self):
        print("\ntest_manyvars_math_inference")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        sample_math_manyvars(basic_vec)
        sample_math_manyvars(basic_vec)
        weld_output = sample_math_manyvars(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([7., 1., 6.]))

    def test_in_out_type_inference(self):
        print("\ntest_in_out_type_inference")
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int32)
        sample_math_ints(basic_vec)
        sample_math_ints(basic_vec)
        weld_output = sample_math_ints(basic_vec)
        numpy.testing.assert_equal(weld_output,
                                   numpy.array([2, 1, 6], dtype=numpy.int32))
        numpy.testing.assert_equal(weld_output.dtype, numpy.int32)

    def test_mixed_types_inference(self):
        print("\ntest_mixed_types")
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int32)
        sample_math_mixed(basic_vec)
        sample_math_mixed(basic_vec)
        weld_output = sample_math_mixed(basic_vec)
        numpy.testing.assert_equal(weld_output,
                                   numpy.array([4.1, 2, 20], dtype=numpy.float64))
        numpy.testing.assert_equal(weld_output.dtype, numpy.float64)

    def test_string_split(self):
        print("\ntest_string_split")
        sample_string: str = "cat dog \n house "
        sample_string_split(sample_string)
        sample_string_split(sample_string)
        weld_output = sample_string_split(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house"])

    def test_string_lower(self):
        print("\ntest_string_lower")
        sample_string: str = "cAt dOg \n houSe. "
        sample_string_lower(sample_string)
        sample_string_lower(sample_string)
        weld_output = sample_string_lower(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house."])

    def test_string_remove_char(self):
        print("\ntest_string_remove_char")
        sample_string: str = "ca..t do..g \n hous...e. "
        sample_string_remove_char(sample_string)
        sample_string_remove_char(sample_string)
        weld_output = sample_string_remove_char(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house"])

    def test_scalar_append(self):
        print("\ntest_scalar_append")
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int64)
        sample_scalar_append(basic_vec, 5)
        sample_scalar_append(basic_vec, 5)
        weld_output = sample_scalar_append(basic_vec, 5)
        output_vec = numpy.array([1, 2, 3, 5], dtype=numpy.int64)
        numpy.testing.assert_equal(weld_output, output_vec)
        self.assertEqual(weld_output.dtype, output_vec.dtype)


if __name__ == '__main__':
    unittest.main()
