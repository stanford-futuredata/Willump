import unittest
import numpy
import importlib

import willump.evaluation.willump_executor as wexec
import willump.graph.willump_graph_node as wgn
import willump.graph.willump_graph as wg
import willump.graph.willump_output_node as wgon
import willump.graph.willump_input_node as wgin
import willump.graph.transform_math_node as wtmn
import willump.evaluation.willump_weld_generator
from willump.graph.transform_math_node import MathOperationInput
from weld.types import *

from willump import pprint_weld

all_floats_type_map = {"__willump_arg0": WeldVec(WeldDouble()), "input": WeldVec(WeldDouble()),
            "output": WeldVec(WeldDouble()),
            "__willump_retval": WeldVec(WeldDouble())}


class BasicEvaluationTests(unittest.TestCase):
    def test_add_literals(self):
        print("\ntest_add_literals")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        add_literal = wtmn.MathOperation("+", "result",
                                         WeldVec(WeldDouble()), MathOperationInput(WeldDouble(),
                                                                                input_literal=1.),
                                         MathOperationInput(WeldDouble(), input_literal=2.))
        add_node: wgn.WillumpGraphNode = wtmn.TransformMathNode([in_node],
                                                                [add_literal, add_literal],
                                                                "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        module_name = wexec.compile_weld_program(weld_program, all_floats_type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([3., 3.]))

    def test_add_indexes(self):
        print("\ntest_add_indexes")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        index_op_one = wtmn.MathOperation("+", "result",
                                          WeldVec(WeldDouble()),
                                        MathOperationInput(WeldDouble(), input_index=("input", 0)),
                                        MathOperationInput(WeldDouble(), input_index=("input", 0)))
        index_op_two = wtmn.MathOperation("+", "result",
                WeldVec(WeldDouble()), MathOperationInput(WeldDouble(), input_index=("input", 1)),
                                          MathOperationInput(WeldDouble(), input_index=("input", 2)))
        index_op_three = wtmn.MathOperation("+", "result",
                                            WeldVec(WeldDouble()), MathOperationInput(WeldDouble(), input_index=("input", 1)),
                                            MathOperationInput(WeldDouble(), input_literal=5.))
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode([in_node],
                                   [index_op_one, index_op_two, index_op_three], "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        module_name = wexec.compile_weld_program(weld_program, all_floats_type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 5., 7.]))

    def test_add_vars(self):
        print("\ntest_add_vars")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        index_op_one = wtmn.MathOperation("+", "two", WeldDouble(),
                                          MathOperationInput(WeldDouble(), input_index=("input", 0)),
                                          MathOperationInput(WeldDouble(), input_index=("input", 0)))
        index_op_two = wtmn.MathOperation("+", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldDouble(), input_index=("input", 1)),
                                          MathOperationInput(WeldDouble(), input_index=("input", 2)))
        index_op_three = wtmn.MathOperation("+", "result", WeldVec(WeldDouble()),
                                            MathOperationInput(WeldDouble(), input_var="two"),
                                            MathOperationInput(WeldDouble(), input_literal=5.))
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode([in_node], [index_op_one, index_op_two, index_op_three],
                                   "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        module_name = wexec.compile_weld_program(weld_program, all_floats_type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([5., 7.]))

    def test_sqrt(self):
        print("\ntest_sqrt")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        index_op_one = wtmn.MathOperation("sqrt", "two", WeldDouble(),
                                          MathOperationInput(WeldDouble(), input_literal=4.))
        index_op_two = wtmn.MathOperation("sqrt", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldDouble(), input_index=("input", 0)))
        index_op_three = wtmn.MathOperation("+", "result", WeldVec(WeldDouble()),
                                            MathOperationInput(WeldDouble(), input_literal=5.),
                                            MathOperationInput(WeldDouble(), input_var="two"))
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode([in_node], [index_op_one, index_op_two, index_op_three],
                                   "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        module_name = wexec.compile_weld_program(weld_program, all_floats_type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([1., 7.]))

    def test_mixed_binops(self):
        print("\ntest_mixed_binops")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        index_op_one = wtmn.MathOperation("-", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldDouble(), input_index=("input", 0)),
                                          MathOperationInput(WeldDouble(), input_index=("input", 0)))
        index_op_two = wtmn.MathOperation("*", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldDouble(), input_index=("input", 1)),
                                          MathOperationInput(WeldDouble(), input_index=("input", 2)))
        index_op_three = wtmn.MathOperation("/", "result", WeldVec(WeldDouble()),
                                            MathOperationInput(WeldDouble(), input_index=("input", 1)),
                                            MathOperationInput(WeldDouble(), input_literal=5.))
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode([in_node], [index_op_one, index_op_two, index_op_three],
                                   "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        module_name = wexec.compile_weld_program(weld_program, all_floats_type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([0., 6., 2./5.]))

    def test_mixed_types(self):
        print("\ntest_mixed_types")
        type_map = {"__willump_arg0": WeldVec(WeldInt()),
                               "input": WeldVec(WeldInt()),
                               "output": WeldVec(WeldDouble()),
                               "__willump_retval": WeldVec(WeldDouble())}
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode("input")
        index_op_one = wtmn.MathOperation("-", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldInt(),
                                                             input_index=("input", 0)),
                                          MathOperationInput(WeldInt(),
                                                             input_index=("input", 0)))
        index_op_two = wtmn.MathOperation("*", "result", WeldVec(WeldDouble()),
                                          MathOperationInput(WeldInt(),
                                                             input_index=("input", 1)),
                                          MathOperationInput(WeldInt(),
                                                             input_index=("input", 2)))
        index_op_three = wtmn.MathOperation("/", "result", WeldVec(WeldDouble()),
                                            MathOperationInput(WeldInt(),
                                                               input_index=("input", 1)),
                                            MathOperationInput(WeldDouble(), input_literal=5.0))
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode([in_node], [index_op_one, index_op_two, index_op_three],
                                   "output", WeldDouble())
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph(out_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                    ["input"], [])

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.int32)
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([0., 6., 2. / 5.]))


if __name__ == '__main__':
    unittest.main()
