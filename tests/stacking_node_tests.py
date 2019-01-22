import unittest
import numpy
import importlib
import inspect
import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import re

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
from sklearn.feature_extraction.text import TfidfVectorizer
from willump import pprint_weld
from willump.graph.willump_graph import WillumpGraph
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
import willump.evaluation.willump_executor as wexec
import scipy.sparse.csr
import scipy.sparse

from typing import Mapping, MutableMapping, List, Tuple
import typing
from weld.types import *

willump_typing_map: MutableMapping[str, WeldType]
willump_static_vars: Mapping[str, object]

with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=0.005, max_df=1.0,
                             lowercase=False, stop_words=None, binary=False, decode_error='replace',
                             vocabulary=simple_vocab_dict)


def sample_stack_sparse(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    transformed_result = scipy.sparse.hstack([transformed_result, transformed_result], format="csr")
    return transformed_result


model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)


def stack_sparse_then_linear_regression(array_one, array_two, input_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = input_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    predicted_result = model.predict(combined_result)
    return predicted_result


tf_idf_vec = \
    TfidfVectorizer(analyzer='char', ngram_range=(2, 5), vocabulary=simple_vocab_dict,
                    lowercase=False)
tf_idf_vec.fit(["theaancatdog house", "bobthe builder", "dogisgooddog"])


def stack_sparse_tfidf(array_one, array_two, input_vect, tf_idf_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = tf_idf_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    return combined_result


def stack_sparse_then_linear_regression_tfidf(array_one, array_two, input_vect, tf_idf_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = tf_idf_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    predicted_result = model.predict(combined_result)
    return predicted_result


class StackingNodeTests(unittest.TestCase):
    def setUp(self):
        global willump_typing_map, willump_static_vars
        willump_typing_map = {}
        willump_static_vars = {}

    def set_typing_map(self, sample_python: str, func_name: str, basic_vec, batch=True) -> None:
        type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery(batch=batch)
        # Create an instrumented AST that will fill willump_typing_map with the Weld types
        # of all variables in the function.
        python_ast = ast.parse(sample_python, mode="exec")
        new_ast: ast.AST = type_discover.visit(python_ast)
        new_ast = ast.fix_missing_locations(new_ast)
        local_namespace = {}
        # Run the instrumented function.
        exec(compile(new_ast, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        result = local_namespace[func_name](*basic_vec)

    def test_sparse_stacking(self):
        print("\ntest_sparse_stacking")
        sample_python: str = inspect.getsource(sample_stack_sparse)
        string_array = ["theaancatdog house", "bobthe builder"]
        transformed_result = vectorizer.transform(string_array)
        correct_result = scipy.sparse.hstack([transformed_result, transformed_result], format="csr").toarray()
        self.set_typing_map(sample_python, "sample_stack_sparse", [string_array, vectorizer])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph, willump_typing_map)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        row, col, data, l, w = local_namespace["sample_stack_sparse"](string_array, vectorizer)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, correct_result)

    def test_sparse_stacking_linear_model(self):
        print("\ntest_sparse_stacking_linear_model")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        sample_python: str = inspect.getsource(stack_sparse_then_linear_regression)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the"]
        result_one = vectorizer.transform(array_one)
        result_two = vectorizer.transform(array_two)
        correct_result = model.predict(scipy.sparse.hstack([result_one, result_two], format="csr"))
        self.set_typing_map(sample_python, "stack_sparse_then_linear_regression", [array_one, array_two, vectorizer])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph, willump_typing_map)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output = local_namespace["stack_sparse_then_linear_regression"](array_one, array_two, vectorizer)
        numpy.testing.assert_equal(weld_output, correct_result)

    def test_sparse_stacking_tfidf(self):
        print("\ntest_sparse_stacking_tfidf")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        sample_python: str = inspect.getsource(stack_sparse_tfidf)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the"]
        result_one = vectorizer.transform(array_one)
        result_two = tf_idf_vec.transform(array_two)
        correct_result = scipy.sparse.hstack([result_one, result_two], format="csr").toarray()
        self.set_typing_map(sample_python, "stack_sparse_tfidf",
                            [array_one, array_two, vectorizer, tf_idf_vec])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph, willump_typing_map)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        row, col, data, l, w =\
            local_namespace["stack_sparse_tfidf"](array_one, array_two, vectorizer, tf_idf_vec)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, correct_result)

    def test_sparse_stacking_linreg_tfidf(self):
        print("\ntest_sparse_stacking_linreg_tfidf")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        sample_python: str = inspect.getsource(stack_sparse_then_linear_regression_tfidf)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the"]
        result_one = vectorizer.transform(array_one)
        result_two = tf_idf_vec.transform(array_two)
        correct_result = model.predict(scipy.sparse.hstack([result_one, result_two], format="csr"))
        self.set_typing_map(sample_python, "stack_sparse_then_linear_regression_tfidf",
                            [array_one, array_two, vectorizer, tf_idf_vec])
        graph_builder: WillumpGraphBuilder = WillumpGraphBuilder(willump_typing_map,
                                                                 willump_static_vars)
        graph_builder.visit(ast.parse(sample_python))
        willump_graph: WillumpGraph = graph_builder.get_willump_graph()
        python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
            willump.evaluation.willump_weld_generator.graph_to_weld(willump_graph, willump_typing_map)
        python_statement_list, modules_to_import = wexec.py_weld_program_to_statements(python_weld_program,
                                                                                       graph_builder.get_aux_data(),
                                                                                       willump_typing_map)
        compiled_functiondef = wexec.py_weld_statements_to_ast(python_statement_list, ast.parse(sample_python))
        for module in modules_to_import:
            globals()[module] = importlib.import_module(module)
        local_namespace = {}
        exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), globals(),
             local_namespace)
        weld_output =\
            local_namespace["stack_sparse_then_linear_regression_tfidf"](array_one, array_two, vectorizer, tf_idf_vec)
        numpy.testing.assert_equal(weld_output, correct_result)
