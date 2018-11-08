import unittest

from willump.inference.python_to_graph import willump_execute_python


class WillumpExecutionTests(unittest.TestCase):
    def test_basic_execution(self):
        print("\ntest_basic_execution")
        with open("tests/test_resources/execution_example.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_execute_python(sample_python)


if __name__ == '__main__':
    unittest.main()
