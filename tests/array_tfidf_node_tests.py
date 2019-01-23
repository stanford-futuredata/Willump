import unittest
import importlib
import numpy
import scipy.sparse
import ast
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from weld.types import *
from typing import List, Tuple
import typing
import inspect
import sklearn.linear_model
import re

with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
tf_idf_vec_char = \
    TfidfVectorizer(analyzer='char', ngram_range=(2, 5), vocabulary=simple_vocab_dict,
                    lowercase=False)

tf_idf_vec_word = \
    TfidfVectorizer(analyzer='word', ngram_range=(1, 3), vocabulary=simple_vocab_dict,
                    lowercase=False)


def sample_tfidf_vectorizer(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)
model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4, -1.5, 0.6]], dtype=numpy.float64)


def tf_idf_vectorizer_then_regression(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    prediction = model.predict(transformed_result)
    return prediction


class TfidfNodeTests(unittest.TestCase):
    def setUp(self):
        global willump_typing_map, willump_static_vars
        willump_typing_map = {}
        willump_static_vars = {}
        self.input_str = [" catdogcat", "the dog house ", "elephantcat", "Bbbbbbb", "an ox cat house", "ananan",
                          "      an ox cat house   ",
                          "Silicone Electric Facial Cleansing Brush Rechargeable Face Cleaning Brush for Face SPA Skin Care Face massage"
                          ]
        tf_idf_vec_char.fit(self.input_str)
        tf_idf_vec_word.fit(self.input_str)
        self.idf_vec = tf_idf_vec_char.idf_
        self.correct_output = tf_idf_vec_char.transform(self.input_str).toarray()
        self.correct_output_word = tf_idf_vec_word.transform(self.input_str).toarray()

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
        local_namespace[func_name](*basic_vec)

    def test_basic_tfidf(self):
        print("\ntest_basic_tfidf")
        aux_data = []
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        array_cv_node: ArrayTfIdfNode = \
            ArrayTfIdfNode(input_node, output_name='lowered_output_words', input_idf_vector=self.idf_vec,
                           input_vocab_dict=simple_vocab_dict,
                           aux_data=aux_data, ngram_range=(2, 5))
        output_node: WillumpOutputNode = WillumpOutputNode(array_cv_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldCSR((WeldDouble())),
                    "__willump_retval0": WeldCSR((WeldDouble()))}
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                                 ["input_str"], aux_data)
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        row, col, data, l, w = weld_llvm_caller.caller_func(self.input_str)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, self.correct_output)

    def test_infer_tfidf_vectorizer(self):
        print("\ntest_infer_tfidf_vectorizer")
        sample_python: str = inspect.getsource(sample_tfidf_vectorizer)
        self.set_typing_map(sample_python, "sample_tfidf_vectorizer", [self.input_str, tf_idf_vec_char])
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
        row, col, data, l, w = local_namespace["sample_tfidf_vectorizer"](self.input_str, tf_idf_vec_char)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, self.correct_output)

    def test_infer_tfidf_vectorizer_word(self):
        print("\ntest_infer_tfidf_vectorizer_word")
        sample_python: str = inspect.getsource(sample_tfidf_vectorizer)
        self.set_typing_map(sample_python, "sample_tfidf_vectorizer", [self.input_str, tf_idf_vec_word])
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
        row, col, data, l, w = local_namespace["sample_tfidf_vectorizer"](self.input_str, tf_idf_vec_word)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, self.correct_output_word)

    def test_linear_model_tfidf_vectorizer(self):
        print("\ntest_linear_model_tfidf_vectorizer")
        sample_python: str = inspect.getsource(tf_idf_vectorizer_then_regression)
        self.set_typing_map(sample_python, "tf_idf_vectorizer_then_regression", [self.input_str, tf_idf_vec_char])
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
        preds = local_namespace["tf_idf_vectorizer_then_regression"](self.input_str, tf_idf_vec_char)
        correct_preds = model.predict(self.correct_output)
        numpy.testing.assert_almost_equal(preds, correct_preds)

    def test_linear_model_tfidf_vectorizer_word(self):
        print("\ntest_linear_model_tfidf_vectorizer_word")
        sample_python: str = inspect.getsource(tf_idf_vectorizer_then_regression)
        self.set_typing_map(sample_python, "tf_idf_vectorizer_then_regression", [self.input_str, tf_idf_vec_word])
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
        preds = local_namespace["tf_idf_vectorizer_then_regression"](self.input_str, tf_idf_vec_word)
        correct_preds = model.predict(self.correct_output_word)
        numpy.testing.assert_almost_equal(preds, correct_preds)
