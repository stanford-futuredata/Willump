import unittest
import numpy
import os
import importlib

import willump.inference.willump_executor as wexec


class WeldLLVMCallerTests(unittest.TestCase):
    def tearDown(self):
        os.remove("code-llvm-opt.ll")

    def test_basic_weld_llvm_caller(self):
        print("\ntest_basic_weld_llvm_caller")
        basic_vec = numpy.array([1., 2., 3.], dtype=numpy.float64)
        # Add 1 to every element in an array.
        weld_program = "(map({0}, |e| e + 1.0))"
        module_name = wexec.compile_weld_program(weld_program)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func(basic_vec)
        numpy.testing.assert_almost_equal(weld_output, numpy.array([2., 3., 4.]))


if __name__ == '__main__':
    unittest.main()
