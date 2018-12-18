import unittest
import numpy
import importlib

import willump.evaluation.willump_executor as wexec

from weld.types import *


class WeldLLVMCallerTests(unittest.TestCase):
    def test_basic_weld_llvm_caller(self):
        print("\ntest_basic_weld_llvm_caller")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        # Add 1 to every element in an array.
        weld_program = "(map(_inp0, |e| e + 1.0))"
        type_map = {"__willump_arg0": WeldVec(WeldDouble()), "e": WeldVec(WeldDouble()),
                    "__willump_retval": WeldVec(WeldDouble())}
        module_name = wexec.compile_weld_program(weld_program, type_map)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 3., 4.]))


if __name__ == '__main__':
    unittest.main()
