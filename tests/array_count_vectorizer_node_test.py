import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from weld.types import *


class ArrayCountVectorizerNodeTests(unittest.TestCase):
    def test_basic_cv(self):
        print("\ntest_basic_cv")
        input_str = ["catcatcat", "dogdogdog", "elephantcat"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
            simple_vocab_dict = {word: index for index, word in
                                 enumerate(simple_vocab.read().splitlines())}
        array_cv_node: ArrayCountVectorizerNode = \
            ArrayCountVectorizerNode(input_node, output_name='lowered_output_words',
                                     input_vocab_dict=simple_vocab_dict,
                                     aux_data=aux_data, ngram_range=(2, 5))
        output_node: WillumpOutputNode = WillumpOutputNode(array_cv_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                                 ["input_str"], aux_data)
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "__willump_retval0": WeldCSR((WeldLong()))}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        numpy.testing.assert_equal(
            weld_output[0], numpy.array([0, 1, 2, 2], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[1], numpy.array([3, 4, 2, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[2], numpy.array([3, 3, 1, 1], dtype=numpy.int64))
