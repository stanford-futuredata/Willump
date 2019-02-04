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
            ArrayCountVectorizerNode(input_node, "input_str", output_name='lowered_output_words',
                                     input_vocab_dict=simple_vocab_dict,
                                     aux_data=aux_data, ngram_range=(2, 5))
        output_node: WillumpOutputNode = WillumpOutputNode(array_cv_node, ["lowered_output_words"])
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldCSR((WeldLong()))}
        weld_output = wexec.execute_from_basics(graph,
                                                type_map,
                                                (input_str,), ["input_str"], ["lowered_output_words"], aux_data)
        numpy.testing.assert_equal(
            weld_output[0], numpy.array([0, 1, 2], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[1], numpy.array([3, 4, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[2], numpy.array([3, 3, 1], dtype=numpy.int64))
