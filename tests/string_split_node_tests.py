import unittest
import numpy
import importlib

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_basic_string_split(self):
        print("\ntest_basic_string_split")
        input_str = "a b c"
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(string_split_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldStr())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["a", "b", "c"])

    def test_char_runs_string_split(self):
        print("\ntest_char_runs_string_split")
        input_str = "   cat    dog    house   "
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(string_split_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldStr())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["cat", "dog", "house"])

    def test_mixed_whitespace_string_split(self):
        print("\ntest_basic_string_split")
        input_str = " \n  c44t   \v d0g  \t\r  elephant123,.  \n "
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(string_split_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldStr())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["c44t", "d0g", "elephant123,."])
