import unittest
import numpy
import importlib
import inspect
import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
from willump import pprint_weld
from willump.graph.willump_graph import WillumpGraph
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
import willump.evaluation.willump_executor as wexec
import scipy.sparse.csr

from typing import Mapping, MutableMapping, List, Tuple
import typing
from weld.types import *

willump_typing_map: MutableMapping[str, WeldType]
willump_static_vars: Mapping[str, object]


def sample_pandas_unpacked(array_one, array_two):
    df = pd.DataFrame()
    df["a"] = array_one
    df["b"] = array_two
    df_a = df["a"].values
    df_b = df["b"].values
    sum_var = df_a + df_b
    mul_sum_var = sum_var * sum_var
    df["ret"] = mul_sum_var
    return df


def sample_pandas_unpacked_nested_ops(array_one, array_two, array_three):
    df = pd.DataFrame()
    df["a"] = array_one
    df["b"] = array_two
    df["c"] = array_three
    df_a = df["a"].values
    df_b = df["b"].values
    df_c = df["c"].values
    op_var = (df_a - df_c) + df_b * df_c
    df["ret"] = op_var
    return df


with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 6), min_df=0.005, max_df=1.0,
                       lowercase=False, stop_words=None, binary=False, decode_error='replace',
                       vocabulary=simple_vocab_dict)


def sample_pandas_count_vectorizer(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


def sample_pandas_merge(left, right):
    new = left.merge(right, how="inner", on="join_column")
    return new


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
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["sample_pandas_unpacked"](vec_one, vec_two)
        numpy.testing.assert_almost_equal(weld_output["ret"].values, numpy.array([25., 49., 81.]))

    def test_unpacked_pandas_nested_ops(self):
        print("\ntest_unpacked_pandas_nested_ops")
        sample_python: str = inspect.getsource(sample_pandas_unpacked_nested_ops)
        vec_one = numpy.array([1., 2., 3.], dtype=numpy.float64)
        vec_two = numpy.array([4., 5., 6.], dtype=numpy.float64)
        vec_three = numpy.array([0., 1., 0.], dtype=numpy.float64)
        self.set_typing_map(sample_python, "sample_pandas_unpacked_nested_ops", [vec_one, vec_two, vec_three])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["sample_pandas_unpacked_nested_ops"](vec_one, vec_two, vec_three)
        numpy.testing.assert_almost_equal(weld_output["ret"].values, numpy.array([1., 6., 3.]))

    def test_pandas_count_vectorizer(self):
        print("\ntest_pandas_count_vectorizer")
        sample_python: str = inspect.getsource(sample_pandas_count_vectorizer)
        string_array = ["theaancatdog house", "bobthe builder"]
        self.set_typing_map(sample_python, "sample_pandas_count_vectorizer", [string_array, vectorizer])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["sample_pandas_count_vectorizer"](string_array, vectorizer)
        numpy.testing.assert_almost_equal(weld_output[0], numpy.array([0, 0, 0, 0, 1]))
        numpy.testing.assert_almost_equal(weld_output[1], numpy.array([0, 4, 3, 5, 0]))
        numpy.testing.assert_almost_equal(weld_output[2], numpy.array([1, 1, 1, 1, 1]))

    def test_pandas_join(self):
        print("\ntest_pandas_join")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        sample_python: str = inspect.getsource(sample_pandas_merge)
        self.set_typing_map(sample_python, "sample_pandas_merge", [left_table, right_table])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["sample_pandas_merge"](left_table, right_table)
        numpy.testing.assert_equal(
            weld_output[1], numpy.array([4, 5, 2, 5, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[3], numpy.array([1.2, 2.2, 2.2, 3.2, 1.2], dtype=numpy.float64))
        numpy.testing.assert_equal(
            weld_output[4], numpy.array([1.3, 2.3, 2.3, 3.3, 1.3], dtype=numpy.float64))
