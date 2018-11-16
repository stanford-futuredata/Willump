import unittest
import numpy
import importlib
import inspect
import ast
import math

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
from willump import pprint_weld
from willump.graph.willump_graph import WillumpGraph

import willump.evaluation.willump_executor as wexec

from typing import Mapping
from weld.types import *


willump_typing_map: Mapping[str, WeldType]


def sample_math_basic(input_numpy_array):
    return_numpy_array = numpy.zeros(3)
    return_numpy_array[0] = 1. + 1.
    return_numpy_array[1] = input_numpy_array[0] - 5.
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


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


def sample_math_ints(input_numpy_array):
    return_numpy_array = numpy.zeros(3, dtype=numpy.int32)
    return_numpy_array[0] = input_numpy_array[0] + input_numpy_array[0]
    return_numpy_array[1] = input_numpy_array[2] - input_numpy_array[1]
    return_numpy_array[2] = input_numpy_array[1] * input_numpy_array[2]
    return return_numpy_array


class GraphInferenceTests(unittest.TestCase):
    def setUp(self):
        global willump_typing_map
        willump_typing_map = {}

    def set_typing_map(self, sample_python: str, func_name: str, basic_vec) -> None:
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
        # Create an instrumented AST that will fill willump_typing_map with the Weld types
        # of all variables in the function.
        python_ast = ast.parse(sample_python, mode="exec")
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        local_namespace = {}
        # Run the instrumented function.
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        local_namespace[func_name](basic_vec)

    def test_basic_math_inference(self):
        print("\ntest_basic_math_inference")
        sample_python: str = inspect.getsource(sample_math_basic)
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        self.set_typing_map(sample_python, "sample_math_basic", basic_vec)
        willump_graph: WillumpGraph = wexec._infer_graph(sample_python)
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., -4., 6.]))

    def test_manyvars_math_inference(self):
        print("\ntest_manyvars_math_inference")
        sample_python: str = inspect.getsource(sample_math_manyvars)
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        self.set_typing_map(sample_python, "sample_math_manyvars", basic_vec)
        willump_graph: WillumpGraph = wexec._infer_graph(sample_python)
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([7., 1., 6.]))

    def test_in_out_type_inference(self):
        print("\ntest_in_out_type_inference")
        sample_python: str = inspect.getsource(sample_math_ints)
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int32)
        self.set_typing_map(sample_python, "sample_math_ints", basic_vec)
        willump_graph: WillumpGraph = wexec._infer_graph(sample_python)
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_equal(weld_output,
                                          numpy.array([2, 1, 6], dtype=numpy.int32))
        numpy.testing.assert_equal(weld_output.dtype, numpy.int32)


if __name__ == '__main__':
    unittest.main()
