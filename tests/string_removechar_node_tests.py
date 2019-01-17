import unittest
import importlib

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_removechar_node import StringRemoveCharNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_basic_char_remove(self):
        print("\ntest_basic_char_remove")
        input_str = ["...a.a", ".b", "c"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        char_remove_node: StringRemoveCharNode = \
            StringRemoveCharNode(input_node, '.', "lowered_output_words")
        output_node: WillumpOutputNode = WillumpOutputNode(char_remove_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "input_str": WeldVec(WeldStr()),
                    "__willump_retval0": WeldVec(WeldStr())}
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], [])
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["aa", "b", "c"])
