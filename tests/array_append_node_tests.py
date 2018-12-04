import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_append_node import ArrayAppendNode
from weld.types import *


class ArrayAppendNodeTests(unittest.TestCase):
    def test_basic_array_append(self):
        print("\ntest_basic_array_append")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        vec_input_node: WillumpInputNode = WillumpInputNode("input_vec")
        val_input_node: WillumpInputNode = WillumpInputNode("input_val")
        array_append_node: ArrayAppendNode = ArrayAppendNode(vec_input_node, val_input_node,
                                                             "output", WeldVec(WeldDouble()))
        output_node: WillumpOutputNode = WillumpOutputNode(array_append_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program: str = willump.evaluation.willump_weld_generator.graph_to_weld(graph)
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_vec", "input_val"], [])
        type_map = {"__willump_arg0": WeldVec(WeldDouble()),
                    "__willump_arg1": WeldLong(),
                    "__willump_retval":  WeldVec(WeldDouble())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec, 5)
        real_output_vec = numpy.array([1., 2., 3., 5.], dtype=numpy.float64)
        numpy.testing.assert_almost_equal(weld_output, real_output_vec)
