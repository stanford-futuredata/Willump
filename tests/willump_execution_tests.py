import unittest
import sys
import numpy
from io import StringIO

from willump.inference.python_to_graph import willump_execute_python
import willump.evaluation.evaluator as weval


class WillumpExecutionTests(unittest.TestCase):
    def tearDown(self):
        weval._weld_object = None

    def test_execution_speed(self):
        print("\ntest_execution_speed")
        with open("tests/test_resources/execution_example.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)

    def test_execution_correctness(self):
        print("\ntest_execution_correctness")
        with open("tests/test_resources/execution_correctness.py", "r") as sample_file:
            saved_stdout = sys.stdout
            sys.stdout = StringIO()
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)
            python_output = sys.stdout.getvalue()
            sys.stdout = saved_stdout
            correct_output = str(numpy.array([2, -4, 6], dtype=numpy.float64)) + "\n" + \
                             str(numpy.array([2, -1, 30], dtype=numpy.float64)) + "\n"
            self.assertEqual(python_output, correct_output)


if __name__ == '__main__':
    unittest.main()
