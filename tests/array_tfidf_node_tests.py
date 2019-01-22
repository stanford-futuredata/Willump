import unittest
import importlib
import numpy
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from weld.types import *


class ArrayCountVectorizerNodeTests(unittest.TestCase):
    def test_basic_tfidf(self):
        print("\ntest_basic_tfidf")
        input_str = ["catdogcat", "thedoghouse", "elephantcat"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
            simple_vocab_dict = {word: index for index, word in
                                 enumerate(simple_vocab.read().splitlines())}
        tf_idf_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), vocabulary=simple_vocab_dict)
        tf_idf_vec.fit(input_str)
        idf_vec = tf_idf_vec.idf_
        correct_output = tf_idf_vec.transform(input_str).toarray()
        array_cv_node: ArrayTfIdfNode = \
            ArrayTfIdfNode(input_node, output_name='lowered_output_words', input_idf_vector=idf_vec,
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
        row, col, data, l, w = weld_llvm_caller.caller_func(input_str)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, correct_output)

