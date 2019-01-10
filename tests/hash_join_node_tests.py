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
            WillumpHashJoinNode(input_node, "output", "join_column", right_table, aux_data)
        print(aux_data)
