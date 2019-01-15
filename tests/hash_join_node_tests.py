import unittest
import importlib
import numpy
import pandas as pd

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.willump_hash_join_node import WillumpHashJoinNode
from weld.types import *


class HashJoinNodeTests(unittest.TestCase):
    def test_basic_hash_join(self):
        print("\ntest_basic_hash_join")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        input_node: WillumpInputNode = WillumpInputNode("input_table")
        aux_data = []
        hash_join_node: WillumpHashJoinNode = \
            WillumpHashJoinNode(input_node, "output", "join_column", right_table, aux_data,
                                size_left_df=3, join_col_left_index=0, batch=True)
        output_node: WillumpOutputNode = WillumpOutputNode(hash_join_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                                 ["input_table"], aux_data)
        type_map = {"__willump_arg0": WeldPandas([WeldVec(WeldLong()), WeldVec(WeldLong()), WeldVec(WeldLong())]),
                    "__willump_retval0": WeldPandas([WeldVec(WeldLong()), WeldVec(WeldLong()), WeldVec(WeldLong()),
                                                     WeldVec(WeldDouble()), WeldVec(WeldDouble())])}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func((left_table["join_column"].values,
                                                    left_table["data1"].values, left_table["data2"].values))
        numpy.testing.assert_equal(
            weld_output[1], numpy.array([4, 5, 2, 5, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[3], numpy.array([1.2, 2.2, 2.2, 3.2, 1.2], dtype=numpy.float64))
        numpy.testing.assert_equal(
            weld_output[4], numpy.array([1.3, 2.3, 2.3, 3.3, 1.3], dtype=numpy.float64))

    def test_point_hash_join(self):
        print("\ntest_point_hash_join")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        input_node: WillumpInputNode = WillumpInputNode("input_table")
        aux_data = []
        hash_join_node: WillumpHashJoinNode = \
            WillumpHashJoinNode(input_node, "output", "join_column", right_table, aux_data,
                                size_left_df=3, join_col_left_index=0, batch=False)
        output_node: WillumpOutputNode = WillumpOutputNode(hash_join_node)
        graph: WillumpGraph = WillumpGraph(output_node)
        weld_program, _, _ = willump.evaluation.willump_weld_generator.graph_to_weld(graph)[0]
        weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                                 ["input_table"], aux_data)
        type_map = {"__willump_arg0": WeldPandas([WeldLong(), WeldLong(), WeldLong()]),
                    "__willump_retval0": WeldPandas([WeldLong(), WeldLong(), WeldLong(), WeldDouble(), WeldDouble()])}
        module_name = wexec.compile_weld_program(weld_program, type_map, aux_data=aux_data)
        weld_llvm_caller = importlib.import_module(module_name)
        weld_output = weld_llvm_caller.caller_func((left_table["join_column"].values[0],
                                                    left_table["data1"].values[0], left_table["data2"].values[0]))
        self.assertEqual(weld_output, (1, 4, 5, 1.2, 1.3))