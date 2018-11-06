import unittest
import numpy

import willump.evaluation.evaluator as weval
import willump.graph.willump_graph_node as wgn
import willump.graph.willump_graph as wg
import willump.graph.willump_output_node as wgon
import willump.graph.willump_input_node as wgin
import willump.graph.transform_math_node as wtmn

from willump import pprint_weld


class BasicEvaluationTests(unittest.TestCase):
    def test_add_literals(self):
        print("\ntest_add_literals")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        add_literal = wtmn.MathOperation("+", True, first_input_literal=1., second_input_literal=2.)
        add_node: wgn.WillumpGraphNode = wtmn.TransformMathNode(in_node, [add_literal, add_literal])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([3., 3.]))

    def test_add_indexes(self):
        print("\ntest_add_indexes")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        index_op_one = wtmn.MathOperation("+", True, first_input_index=0, second_input_index=0)
        index_op_two = wtmn.MathOperation("+", True, first_input_index=1, second_input_index=2)
        index_op_three = wtmn.MathOperation("+", True, first_input_index=1, second_input_literal=5.)
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode(in_node, [index_op_one, index_op_two, index_op_three])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 5., 7.]))

    def test_add_vars(self):
        print("\ntest_add_vars")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        index_op_one = wtmn.MathOperation("+", False, first_input_index=0, second_input_index=0,
                                          store_result_var_name="two")
        index_op_two = wtmn.MathOperation("+", True, first_input_index=1, second_input_index=2,
                                          store_result_var_name="five")
        index_op_three = wtmn.MathOperation("+", True, first_input_var="five",
                                            second_input_literal=5.)
        index_op_four = wtmn.MathOperation("+", True, first_input_var="two", second_input_var="two")
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode(in_node, [index_op_one, index_op_two, index_op_three,
                                             index_op_four])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([5., 10., 4.]))

    def test_sqrt(self):
        print("\ntest_sqrt")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        index_op_one = wtmn.MathOperation("sqrt", False, first_input_literal=4.0,
                                          store_result_var_name="two")
        index_op_two = wtmn.MathOperation("sqrt", True, first_input_index=0)
        index_op_three = wtmn.MathOperation("+", True, first_input_literal=5.0,
                                            second_input_var="two")
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode(in_node, [index_op_one, index_op_two, index_op_three])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([1., 7.]))

    def test_mixed_binops(self):
        print("\ntest_mixed_binops")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        index_op_one = wtmn.MathOperation("-", True, first_input_index=0, second_input_index=0)
        index_op_two = wtmn.MathOperation("*", True, first_input_index=1, second_input_index=2)
        index_op_three = wtmn.MathOperation("/", True, first_input_index=1, second_input_literal=5.)
        add_node: wgn.WillumpGraphNode = \
            wtmn.TransformMathNode(in_node, [index_op_one, index_op_two, index_op_three])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([0., 6., 2./5.]))


if __name__ == '__main__':
    unittest.main()
