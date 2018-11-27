import unittest
import numpy
import ast

from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
from typing import Mapping
from weld.types import *

basic_no_arrays_math = \
"""
def basic_no_arrays_math(c):
    b = 3
    a = 3.
    return a + b + c
basic_no_arrays_math(3)
"""

math_with_arrays = \
"""
def math_with_arrays(input_numpy_array):
    a = 3
    return_numpy_array = numpy.zeros(3, dtype=numpy.float64)
    return_numpy_array[0] = input_numpy_array[0]
    return_numpy_array[1] = input_numpy_array[1] + a
    return_numpy_array[2] = input_numpy_array[2] + 3.
    return return_numpy_array
math_with_arrays(numpy.zeros(3, dtype=numpy.int32))
"""

basic_string_splitting = \
"""
def basic_string_splitting(in_string):
    out_string = in_string.split()
    return out_string
basic_string_splitting("a b c")
"""

willump_typing_map: Mapping[str, WeldType]


class RuntimeTypeDiscoveryTests(unittest.TestCase):

    def setUp(self):
        global willump_typing_map
        willump_typing_map = {}

    def test_basic_type_discovery(self):
        print("\ntest_basic_type_discovery")
        sample_python: str = basic_no_arrays_math
        python_ast: ast.AST = ast.parse(sample_python)
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(), locals())
        self.assertTrue(isinstance(willump_typing_map["c"], WeldInt))
        self.assertTrue(isinstance(willump_typing_map["__willump_arg0"], WeldInt))
        self.assertTrue(isinstance(willump_typing_map["a"], WeldDouble))
        self.assertTrue(isinstance(willump_typing_map["b"], WeldInt))
        self.assertTrue(isinstance(willump_typing_map["__willump_retval"], WeldDouble))

    def test_arrays_type_discovery(self):
        print("\ntest_arrays_type_discovery")
        sample_python: str = math_with_arrays
        python_ast: ast.AST = ast.parse(sample_python)
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(), locals())
        self.assertTrue(isinstance(willump_typing_map["a"], WeldInt))
        self.assertTrue(isinstance(willump_typing_map["input_numpy_array"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["input_numpy_array"].elemType, WeldInt))
        self.assertTrue(isinstance(willump_typing_map["__willump_arg0"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["__willump_arg0"].elemType, WeldInt))
        self.assertTrue(isinstance(willump_typing_map["return_numpy_array"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["return_numpy_array"].elemType, WeldDouble))
        self.assertTrue(isinstance(willump_typing_map["__willump_retval"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["__willump_retval"].elemType, WeldDouble))

    def test_strings_type_discovery(self):
        print("\ntest_strings_type_discovery")
        sample_python: str = basic_string_splitting
        python_ast: ast.AST = ast.parse(sample_python)
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(), locals())
        self.assertTrue(isinstance(willump_typing_map["in_string"], WeldStr))
        self.assertTrue(isinstance(willump_typing_map["__willump_arg0"], WeldStr))
        self.assertTrue(isinstance(willump_typing_map["out_string"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["out_string"].elemType, WeldStr))
        self.assertTrue(isinstance(willump_typing_map["__willump_retval"], WeldVec))
        self.assertTrue(isinstance(willump_typing_map["__willump_retval"].elemType, WeldStr))


if __name__ == '__main__':
    unittest.main()
