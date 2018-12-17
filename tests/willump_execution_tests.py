import unittest
import os
import numpy
import sys


class WillumpExecutionTests(unittest.TestCase):
    def setUp(self):
        if "tests/test_scripts" not in sys.path:
            sys.path.append("tests/test_scripts")

    def test_execution_speed(self):
        print("\ntest_execution_speed")
        import execution_example
        execution_example.main()

    def test_execution_speed_simple(self):
        print("\ntest_execution_speed_simple")
        import execution_example_simple
        execution_example_simple.main()

    def test_execution_veracity(self):
        print("\ntest_execution_veracity")
        import execution_correctness
        execution_correctness.main()
        with open("temp_execution_correctness_file.tmp", "r") as output_file:
            python_output = output_file.read()
        correct_output = str(numpy.array([2, -4, 6], dtype=numpy.float64)) + "\n" + \
                         str(numpy.array([2, -1, 30], dtype=numpy.float64)) + "\n"
        self.assertEqual(python_output, correct_output)
        os.remove("temp_execution_correctness_file.tmp")


if __name__ == '__main__':
    unittest.main()
