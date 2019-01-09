import unittest
import importlib
import numpy

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.vocabulary_frequency_count_node import VocabularyFrequencyCountNode
from willump.graph.logistic_regression_node import LogisticRegressionNode
from weld.types import *


class StringSplitNodeTests(unittest.TestCase):
    def test_binary_logistic_regression1(self):
        print("\ntest_binary_logistic_regression1")
        input_vec = numpy.array([0, 0, 0, 1, 1, 0], dtype=numpy.int64)
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        logistic_regression_weights = numpy.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]],
                                                  dtype=numpy.float64)
        logistic_regression_intercept = numpy.array([1.0],
                                                  dtype=numpy.float64)
        logistic_regression_node: LogisticRegressionNode = \
            LogisticRegressionNode(input_node, "logit_output", logistic_regression_weights,
            logistic_regression_intercept, aux_data)
        output_node: WillumpOutputNode = WillumpOutputNode(logistic_regression_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], aux_data)
        type_map = {"__willump_arg0": WeldVec(WeldLong()),
                    "__willump_retval0": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_vec)
        numpy.testing.assert_equal(
            weld_output, numpy.array([1], dtype=numpy.int64))

    def test_binary_logistic_regression2(self):
        print("\ntest_binary_logistic_regression2")
        input_vec = numpy.array([0, 0, 0, 1, 1, 0], dtype=numpy.int64)
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        aux_data = []
        logistic_regression_weights = numpy.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]],
                                                  dtype=numpy.float64)
        logistic_regression_intercept = numpy.array([0.001],
                                                  dtype=numpy.float64)
        logistic_regression_node: LogisticRegressionNode = \
            LogisticRegressionNode(input_node, "logit_output", logistic_regression_weights,
            logistic_regression_intercept, aux_data)
        output_node: WillumpOutputNode = WillumpOutputNode(logistic_regression_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input_str"], aux_data)
        type_map = {"__willump_arg0": WeldVec(WeldLong()),
                    "__willump_retval0": WeldVec(WeldLong())}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(input_vec)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0], dtype=numpy.int64))

