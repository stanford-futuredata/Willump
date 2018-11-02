import unittest
import willump.evaluation.evaluator as weval
import willump.graph.willump_graph_node as wgn
import willump.graph.willump_graph as wg
import willump.graph.willump_output_node as wgon
import willump.graph.willump_input_node as wgin
import willump.graph.transform_math_node as tmn
import numpy


class BasicEvaluationTests(unittest.TestCase):
    def test_add_literals(self):
        print("\ntest_add_literals")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        basic_mathop = tmn.MathOperation("+", True, first_input_literal=1., second_input_literal=2.)
        add_node: wgn.WillumpGraphNode = tmn.TransformMathNode(in_node, [basic_mathop, basic_mathop])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([3., 3.]))

    def test_add_indexes(self):
        print("\ntest_add_indexes")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        index_op_one = tmn.MathOperation("+", True, first_input_index=0, second_input_index=0)
        index_op_two = tmn.MathOperation("+", True, first_input_index=1, second_input_index=2)
        index_op_three = tmn.MathOperation("+", True, first_input_index=1, second_input_literal=5.)
        add_node: wgn.WillumpGraphNode = \
            tmn.TransformMathNode(in_node, [index_op_one, index_op_two, index_op_three])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 5., 7.]))


if __name__ == '__main__':
    unittest.main()
