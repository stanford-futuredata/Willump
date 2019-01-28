import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_space_combiner_node import ArraySpaceCombinerNode
from weld.types import *


class ArraySpaceCombinerNodeTests(unittest.TestCase):
    def test_basic_combiner(self):
        print("\ntest_basic_combiner")
        input_str = ["catcat  \t  cat", "dogd o g  dog", "elephant         cat"]
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        array_combiner_node: ArraySpaceCombinerNode = \
            ArraySpaceCombinerNode(input_node, output_name='lowered_output_words')
        output_node: WillumpOutputNode = WillumpOutputNode(array_combiner_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldVec(WeldStr())}
        weld_output = wexec.execute_from_basics(graph,
                                                type_map,
                                                (input_str,), ["input_str"], ["lowered_output_words"], aux_data)
        self.assertEqual(weld_output, ["catcat cat", "dogd o g dog", "elephant cat"])
