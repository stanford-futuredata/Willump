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


if __name__ == '__main__':
    unittest.main()
