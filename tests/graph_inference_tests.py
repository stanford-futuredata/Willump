import unittest
import numpy

from willump import pprint_weld
from willump.inference.willump_executor import infer_graph
from willump.graph.willump_graph import WillumpGraph

import willump.evaluation.evaluator as weval


class GraphInferenceTests(unittest.TestCase):
    def tearDown(self):
        weval._weld_object = None

    def test_basic_math_inference(self):
        print("\ntest_basic_math_inference")
        with open("tests/test_resources/sample_math.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_graph: WillumpGraph = infer_graph(sample_python)
            weld_str: str = weval.graph_to_weld(willump_graph)

            basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
            weld_output = weval.evaluate_weld(weld_str, basic_vec)
            numpy.testing.assert_almost_equal(weld_output, numpy.array([2., -4., 6.]))

    def test_manyvars_math_inference(self):
        print("\ntest_manyvars_math_inference")
        with open("tests/test_resources/sample_math_manyvars.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_graph: WillumpGraph = infer_graph(sample_python)
            weld_str: str = weval.graph_to_weld(willump_graph)
            basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
            weld_output = weval.evaluate_weld(weld_str, basic_vec)
            numpy.testing.assert_almost_equal(weld_output, numpy.array([7., 1., 6.]))


if __name__ == '__main__':
    unittest.main()
