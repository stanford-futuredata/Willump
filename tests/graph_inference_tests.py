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
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
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


def sample_math_mixed(input_numpy_array):
    intermediate_numpy_array = numpy.zeros(1, dtype=numpy.float64)
    intermediate_numpy_array[0] = 5 + 5
    three_float = 2 + 1.0
    return_numpy_array = numpy.zeros(3, dtype=numpy.float64)
    return_numpy_array[0] = input_numpy_array[0] + 3.1
    return_numpy_array[1] = input_numpy_array[2] - three_float + math.sqrt(4)
    return_numpy_array[2] = input_numpy_array[1] * intermediate_numpy_array[0]
    return return_numpy_array


def sample_string_split(input_string):
    output_strings = input_string.split()
    return output_strings


def sample_string_lower(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].lower()
    return output_strings


def sample_string_remove_char(input_string):
    output_strings = input_string.split()
    for i in range(len(output_strings)):
        output_strings[i] = output_strings[i].replace(".", "")
    return output_strings


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
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., -4., 6.]))

    def test_manyvars_math_inference(self):
        print("\ntest_manyvars_math_inference")
        sample_python: str = inspect.getsource(sample_math_manyvars)
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        self.set_typing_map(sample_python, "sample_math_manyvars", basic_vec)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([7., 1., 6.]))

    def test_in_out_type_inference(self):
        print("\ntest_in_out_type_inference")
        sample_python: str = inspect.getsource(sample_math_ints)
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int32)
        self.set_typing_map(sample_python, "sample_math_ints", basic_vec)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_equal(weld_output,
                                          numpy.array([2, 1, 6], dtype=numpy.int32))
        numpy.testing.assert_equal(weld_output.dtype, numpy.int32)

    def test_mixed_types_inference(self):
        print("\ntest_mixed_types")
        sample_python: str = inspect.getsource(sample_math_mixed)
        basic_vec = numpy.array([1, 2, 3], dtype=numpy.int32)
        self.set_typing_map(sample_python, "sample_math_mixed", basic_vec)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_equal(weld_output,
                                          numpy.array([4.1, 2, 20], dtype=numpy.float64))
        numpy.testing.assert_equal(weld_output.dtype, numpy.float64)

    def test_string_split(self):
        print("\ntest_string_split")
        sample_python: str = inspect.getsource(sample_string_split)
        sample_string: str = "cat dog \n house "
        self.set_typing_map(sample_python, "sample_string_split", sample_string)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house"])

    def test_string_lower(self):
        print("\ntest_string_lower")
        sample_python: str = inspect.getsource(sample_string_lower)
        sample_string: str = "cAt dOg \n houSe. "
        self.set_typing_map(sample_python, "sample_string_lower", sample_string)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house."])

    def test_string_remove_char(self):
        print("\ntest_string_remove_char")
        sample_python: str = inspect.getsource(sample_string_remove_char)
        sample_string: str = "ca..t do..g \n hous...e. "
        self.set_typing_map(sample_python, "sample_string_remove_char", sample_string)
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        weld_program: str = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    graph_builder.get_args_list(), graph_builder.get_aux_data())
        module_name = wexec.compile_weld_program(weld_program, willump_typing_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(sample_string)
        numpy.testing.assert_equal(weld_output, ["cat", "dog", "house"])


if __name__ == '__main__':
    unittest.main()
