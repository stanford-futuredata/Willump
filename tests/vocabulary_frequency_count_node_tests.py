import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.vocabulary_frequency_count_node import VocabularyFrequencyCountNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_basic_frequency(self):
        print("\ntest_basic_frequency")
        input_str = "cat dog elephant"
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        vocab_count_node: VocabularyFrequencyCountNode = \
            VocabularyFrequencyCountNode(string_split_node, output_name='lowered_output_words',
                                         input_vocabulary_filename=
                                         "tests/test_resources/simple_vocabulary.txt")
        aux_data = [vocab_count_node.get_aux_data()]
        output_node: WillumpOutputNode = WillumpOutputNode(vocab_count_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0, 0, 0, 1, 1, 0], dtype=numpy.int64))

    def test_multiples_frequency(self):
        print("\ntest_multiples_frequency")
        input_str = "the cat the dog the elephant a dog"
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        vocab_count_node: VocabularyFrequencyCountNode = \
            VocabularyFrequencyCountNode(string_split_node, output_name='lowered_output_words',
                                         input_vocabulary_filename=
                                         "tests/test_resources/simple_vocabulary.txt")
        aux_data = [vocab_count_node.get_aux_data()]
        output_node: WillumpOutputNode = WillumpOutputNode(vocab_count_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        numpy.testing.assert_equal(
            weld_output, numpy.array([3, 1, 0, 1, 2, 0], dtype=numpy.int64))

