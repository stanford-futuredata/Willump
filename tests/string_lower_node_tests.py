import unittest
import importlib

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.string_lower_node import StringLowerNode
from weld.types import *


class StringLowerNodeTests(unittest.TestCase):
    def test_basic_string_lower(self):
        print("\ntest_basic_string_lower")
        input_str = ["aAa", "Bb", "cC"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_lower_node: StringLowerNode =\
            StringLowerNode(input_node, "lowered_output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(string_lower_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldVec(WeldStr())}
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], [])
        module_name = wexec.compile_weld_program(weld_program, type_map, ["input_str"], ["lowered_output_words"])
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output, = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["aaa", "bb", "cc"])

    def test_mixed_string_lower(self):
        print("\ntest_mixed_string_lower")
        input_str = ["aA,.,.a", "B,,b", "c34234C"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_lower_node: StringLowerNode =\
            StringLowerNode(input_node, "lowered_output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(string_lower_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldVec(WeldStr())}
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], [])
        module_name = wexec.compile_weld_program(weld_program, type_map, ["input_str"], ["lowered_output_words"])
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output, = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["aa,.,.a", "b,,b", "c34234c"])

