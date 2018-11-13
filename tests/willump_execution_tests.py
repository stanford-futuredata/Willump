import unittest
import os
import numpy

from willump.inference.willump_executor import willump_execute_python


class WillumpExecutionTests(unittest.TestCase):
    def test_execution_speed(self):
        print("\ntest_execution_speed")
        with open("tests/test_resources/execution_example.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)

    def test_execution_speed_simple(self):
        print("\ntest_execution_speed_simple")
        with open("tests/test_resources/execution_example_simple.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)

    def test_execution_veracity(self):
        print("\ntest_execution_veracity")
        with open("tests/test_resources/execution_correctness.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)
            with open("temp_execution_correctness_file.tmp", "r") as output_file:
                python_output = output_file.read()
            correct_output = str(numpy.array([2, -4, 6], dtype=numpy.float64)) + "\n" + \
                             str(numpy.array([2, -1, 30], dtype=numpy.float64)) + "\n"
            self.assertEqual(python_output, correct_output)
            os.remove("temp_execution_correctness_file.tmp")


if __name__ == '__main__':
    unittest.main()
