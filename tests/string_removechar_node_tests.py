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
        type_map = {"input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldVec(WeldStr())}
        weld_output = wexec.execute_from_basics(graph, type_map, (input_str,), ["input_str"], ["lowered_output_words"],
                                                [])
        self.assertEqual(weld_output, ["aa", "b", "c"])
