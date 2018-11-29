import unittest
import importlib

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.string_removechar_node import StringRemoveCharNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_basic_char_remove(self):
        print("\ntest_basic_char_remove")
        input_str = "...a.a .,b c,,"
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_split_node: StringSplitNode = StringSplitNode(input_node, "output_words")
        char_remove_node: StringRemoveCharNode = \
            StringRemoveCharNode(string_split_node, '.', "lowered_output_words")
        char_remove_node = \
            StringRemoveCharNode(char_remove_node, ',', "lowered_output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(char_remove_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        type_map = {"__willump_arg0": WeldStr(),
                    "__willump_retval": WeldVec(WeldStr())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["aa", "b", "c"])
