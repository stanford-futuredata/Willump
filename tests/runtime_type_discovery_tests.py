import unittest
import numpy
import ast

from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from typing import Mapping

basic_no_arrays_math = \
"""
def basic_no_arrays_math(a):
    b = 3
    return a
basic_no_arrays_math(3)
"""

willump_typing_map = {}


class RuntimeTypeDiscoveryTests(unittest.TestCase):
    def test_basic_type_discovery(self):
        print("\ntest_basic_type_discovery")
        sample_python: str = basic_no_arrays_math
        python_ast: ast.AST = ast.parse(sample_python)
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(), locals())
        print(ast.dump(new_ast))
        print(willump_typing_map)


if __name__ == '__main__':
    unittest.main()
