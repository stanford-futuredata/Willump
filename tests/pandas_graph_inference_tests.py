import unittest
import numpy
import importlib
import inspect
import ast
import pandas as pd

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
from willump import pprint_weld
from willump.graph.willump_graph import WillumpGraph
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
import willump.evaluation.willump_executor as wexec

from typing import Mapping, MutableMapping, List, Tuple
import typing
from weld.types import *
import astpretty

willump_typing_map: MutableMapping[str, WeldType]
willump_static_vars: Mapping[str, object]



def sample_pandas_unpacked(array_one, array_two):
    df = pd.DataFrame()
    df["a"] = array_one
    df["b"] = array_two
    df_a = df["a"].values
    df_b = df["b"].values
    sum_var = df_a + df_b
    df["c"] = sum_var
    return df


class PandasGraphInferenceTests(unittest.TestCase):
    def setUp(self):
        global willump_typing_map, willump_static_vars
        willump_typing_map = {}
        willump_static_vars = {}

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
        local_namespace[func_name](*basic_vec)

    def test_unpacked_pandas(self):
        print("\ntest_unpacked_pandas")
        sample_python: str = inspect.getsource(sample_pandas_unpacked)
        vec_one = numpy.array([1., 2., 3.], dtype=numpy.float64)
        vec_two = numpy.array([4., 5., 6.], dtype=numpy.float64)
        self.set_typing_map(sample_python, "sample_pandas_unpacked", [vec_one, vec_two])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                    graph_builder.get_aux_data(), willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["sample_pandas_unpacked"](vec_one, vec_two)
        numpy.testing.assert_almost_equal(weld_output["c"].values, numpy.array([5., 7., 9.]))
