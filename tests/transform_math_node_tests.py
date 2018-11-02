import unittest
import willump.evaluation.evaluator as weval
import willump.graph.willump_graph_node as wgn
import willump.graph.willump_graph as wg
import willump.graph.willump_output_node as wgon
import willump.graph.willump_input_node as wgin
import willump.graph.transform_math_node as tmn
import numpy


class BasicEvaluationTests(unittest.TestCase):
    def test_all_literals(self):
        print("\nTest evaluate graph")
        in_node: wgin.WillumpInputNode = wgin.WillumpInputNode()
        basic_mathop = tmn.MathOperation("+", True, first_input_literal=1., second_input_literal=2.)
        add_node: wgn.WillumpGraphNode = tmn.TransformMathNode(in_node, [basic_mathop, basic_mathop])
        out_node: wgon.WillumpOutputNode = wgon.WillumpOutputNode(add_node)
        graph: wg.WillumpGraph = wg.WillumpGraph([in_node], out_node)
        weld_str: str = weval.graph_to_weld(graph)

        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        weld_output = weval.evaluate_weld(weld_str, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([3., 3.]))


if __name__ == '__main__':
    unittest.main()
