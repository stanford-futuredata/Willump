import unittest
import numpy
import willump.evaluation.willump_executor as wexec


@wexec.willump_execute()
def sample_basic_udf(input_string):
    output_strings = input_string.upper()
    split_output_strings = output_strings.split()
    return split_output_strings


class WillumpPythonUDFTests(unittest.TestCase):
    def test_basic_udf(self):
        print("\ntest_basic_udf")
        sample_string: str = "cat dog \n house. "
        sample_basic_udf(sample_string)
        sample_basic_udf(sample_string)
        weld_output = sample_basic_udf(sample_string)
        numpy.testing.assert_equal(weld_output, ["CAT", "DOG", "HOUSE."])
