import unittest
import willump.evaluation.evaluator as eval
import willump.graph.willump_graph_node as wgn
import willump.graph.willump_graph as wg
import willump.graph.willump_output_node as wgon
import willump.graph.willump_input_node as wgin
import willump.graph.transform_addition_node as tan
import numpy


class BasicEvaluationTests(unittest.TestCase):
    def test_evaluate_weld(self):
        print("\nTest evaluate weld")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        # Add 1 to every element in an array.
        weld_program = "(map({0}, |e| e + 1.0))"
        weld_output = eval.evaluate_weld(weld_program, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 3., 4.]))

    def test_basic_weld_graph(self):
        print("\nTest evaluate graph")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        add_node: wgn.WillumpGraphNode = tan.TransformAdditionNode(in_node, 1, 2)
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = eval.graph_to_weld(graph)
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = eval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([5.]))


if __name__ == '__main__':
    unittest.main()
