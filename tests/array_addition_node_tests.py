import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_addition_node import ArrayAdditionNode
from weld.types import *


class ArrayAppendNodeTests(unittest.TestCase):
    def test_basic_array_append(self):
        print("\ntest_basic_array_append")
        vec_1 = numpy.array([1.1, 2.2, 3.3], dtype=numpy.float64)
        vec_2 = numpy.array([4, 5, 6], dtype=numpy.int64)
        vec1_input_node: WillumpInputNode = WillumpInputNode("vec_1")
        vec2_input_node: WillumpInputNode = WillumpInputNode("vec_2")
        array_addition_node: ArrayAdditionNode = ArrayAdditionNode(vec1_input_node, vec2_input_node,
                                                "output", WeldVec(WeldDouble()))
        output_node: WillumpOutputNode = WillumpOutputNode(array_addition_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["vec_1", "vec_2"], [])
        type_map = {"__willump_arg0": WeldVec(WeldDouble()),
                    "__willump_arg1": WeldVec(WeldLong()),
                    "__willump_retval":  WeldVec(WeldDouble())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(vec_1, vec_2)
        real_output_vec = numpy.array([5.1, 7.2, 9.3], dtype=numpy.float64)
        numpy.testing.assert_almost_equal(weld_output, real_output_vec)
