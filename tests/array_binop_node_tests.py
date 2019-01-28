import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_binop_node import ArrayBinopNode
from weld.types import *


class ArrayAppendNodeTests(unittest.TestCase):
    def test_basic_array_add(self):
        print("\ntest_basic_array_add")
        vec_1 = numpy.array([1.1, 2.2, 3.3], dtype=numpy.float64)
        vec_2 = numpy.array([4, 5, 6], dtype=numpy.int64)
        vec1_input_node: WillumpInputNode = WillumpInputNode("vec_1")
        vec2_input_node: WillumpInputNode = WillumpInputNode("vec_2")
        array_addition_node: ArrayBinopNode = ArrayBinopNode(vec1_input_node, vec2_input_node,
                                                "output", WeldVec(WeldDouble()), "+")
        output_node: WillumpOutputNode = WillumpOutputNode(array_addition_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"vec_1": WeldVec(WeldDouble()),
                    "vec_2": WeldVec(WeldLong()),
                    "output": WeldVec(WeldDouble())}
        weld_output = wexec.execute_from_basics(graph,
                                                type_map, (vec_1, vec_2), ["vec_1", "vec_2"], ["output"], [])
        real_output_vec = numpy.array([5.1, 7.2, 9.3], dtype=numpy.float64)
        numpy.testing.assert_almost_equal(weld_output, real_output_vec)
