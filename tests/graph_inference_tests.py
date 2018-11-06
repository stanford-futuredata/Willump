import unittest

from willump.inference.python_to_graph import infer_graph


class GraphInferenceTests(unittest.TestCase):
    def test_graph_inference(self):
        print("\ntest_graph_inference")
        with open("tests/test_resources/sample_math.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            infer_graph(sample_python)


if __name__ == '__main__':
    unittest.main()
