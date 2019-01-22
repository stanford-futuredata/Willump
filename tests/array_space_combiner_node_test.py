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
        type_map = {"__willump_arg0": WeldVec(WeldStr()),
                    "input_str": WeldVec(WeldStr()),
                    "lowered_output_words": WeldVec(WeldStr()),
                    "__willump_retval0": WeldVec(WeldStr())}
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                                 ["input_str"], aux_data)
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_str)
        self.assertEqual(weld_output, ["catcat cat", "dogd o g dog", "elephant cat"])
