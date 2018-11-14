import unittest
import numpy
import importlib
import os

import willump.inference.willump_weld_generator
from willump import pprint_weld
from willump.graph.willump_graph import WillumpGraph

import willump.inference.willump_executor as wexec


class GraphInferenceTests(unittest.TestCase):
    def tearDown(self):
        os.remove("code-llvm-opt.ll")

    def test_basic_math_inference(self):
        print("\ntest_basic_math_inference")
        with open("tests/test_resources/sample_math.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_graph: WillumpGraph = wexec.infer_graph(sample_python)
            weld_program: str = willump.inference.willump_weld_generator.graph_to_weld(willump_graph)

            basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
            module_name = wexec.compile_weld_program(weld_program)
            weld_llvm_caller = importlib.import_module(module_name)
            weld_output = weld_llvm_caller.caller_func(basic_vec)
            numpy.testing.assert_almost_equal(weld_output, numpy.array([2., -4., 6.]))

    def test_manyvars_math_inference(self):
        print("\ntest_manyvars_math_inference")
        with open("tests/test_resources/sample_math_manyvars.py", "r") as sample_file:
            sample_python: str = sample_file.read()
            willump_graph: WillumpGraph = wexec.infer_graph(sample_python)
            weld_program: str = willump.inference.willump_weld_generator.graph_to_weld(willump_graph)

            basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
            module_name = wexec.compile_weld_program(weld_program)
            weld_llvm_caller = importlib.import_module(module_name)
            weld_output = weld_llvm_caller.caller_func(basic_vec)
            numpy.testing.assert_almost_equal(weld_output, numpy.array([7., 1., 6.]))


if __name__ == '__main__':
    unittest.main()
