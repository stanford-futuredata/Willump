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
        input_str = ["cat", "dog", "elephant"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
            simple_vocab_dict = {word: index for index, word in
                                 enumerate(simple_vocab.read().splitlines())}
        vocab_count_node: VocabularyFrequencyCountNode = \
            VocabularyFrequencyCountNode(input_node, output_name='lowered_output_words',
                                         input_vocab_dict=simple_vocab_dict,
                                         aux_data=aux_data)
        output_node: WillumpOutputNode = WillumpOutputNode(vocab_count_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], aux_data)
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "__willump_retval0": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0, 0, 0, 1, 1, 0], dtype=numpy.int64))

    def test_multiples_frequency(self):
        print("\ntest_multiples_frequency")
        input_str = "the cat the dog the elephant a dog".split()
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
            simple_vocab_dict = {word: index for index, word in
                                 enumerate(simple_vocab.read().splitlines())}
        vocab_count_node: VocabularyFrequencyCountNode = \
            VocabularyFrequencyCountNode(input_node, output_name='lowered_output_words',
                                         input_vocab_dict=simple_vocab_dict,
                                         aux_data=aux_data)
        output_node: WillumpOutputNode = WillumpOutputNode(vocab_count_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], aux_data)
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "__willump_retval0": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        numpy.testing.assert_equal(
            weld_output, numpy.array([3, 1, 0, 1, 2, 0], dtype=numpy.int64))

