import unittest
import willump.evaluation.evaluator as weval
import numpy


class BasicEvaluationTests(unittest.TestCase):
    def test_evaluate_weld(self):
        print("\nTest evaluate weld")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        # Add 1 to every element in an array.
        weld_program = "(map({0}, |e| e + 1.0))"
        weld_output = weval.evaluate_weld(weld_program, basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 3., 4.]))


if __name__ == '__main__':
    unittest.main()
