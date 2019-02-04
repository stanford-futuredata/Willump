import unittest

import willump.evaluation.willump_executor as wexec

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
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
        weld_output = wexec.execute_from_basics(graph, type_map, (input_str,), ["input_str"], ["lowered_output_words"],
                                                [])
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
        weld_output = wexec.execute_from_basics(graph, type_map, (input_str,), ["input_str"], ["lowered_output_words"],
                                                [])
        self.assertEqual(weld_output, ["aa,.,.a", "b,,b", "c34234c"])

