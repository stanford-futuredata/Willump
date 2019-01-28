import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.linear_regression_node import LinearRegressionNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_binary_logistic_regression_vec_input(self):
        print("\ntest_binary_logistic_regression_vec_input")
        input_vec = numpy.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 3, 0]], dtype=numpy.int64)
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        logistic_regression_weights = numpy.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]],
                                                  dtype=numpy.float64)
        logistic_regression_intercept = numpy.array([1.0],
                                                  dtype=numpy.float64)
        logistic_regression_node: LinearRegressionNode = \
            LinearRegressionNode(input_node, WeldVec(WeldVec(WeldLong())), "logit_output", WeldVec(WeldLong()),
                                 logistic_regression_weights, logistic_regression_intercept, aux_data)
        output_node: WillumpOutputNode = WillumpOutputNode(logistic_regression_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldVec(WeldVec(WeldLong())),
                    "logit_output": WeldVec(WeldLong())}
        weld_output = wexec.execute_from_basics(graph, type_map, (input_vec,), ["input_str"], ["logit_output"],
                                                aux_data)
        numpy.testing.assert_equal(
            weld_output, numpy.array([1, 0], dtype=numpy.int64))
