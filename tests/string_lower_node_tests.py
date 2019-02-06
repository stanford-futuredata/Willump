import unittest

import willump.evaluation.willump_executor as wexec

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_lower_node import StringLowerNode
from weld.types import *

import pandas as pd


class StringLowerNodeTests(unittest.TestCase):
    def test_mixed_string_lower(self):
        print("\ntest_mixed_string_lower")
        input_df = pd.DataFrame({"target_col": ["aA,.,.a", "B,,b", "c34234C"]})
        input_str = list(input_df["target_col"].values)
        input_node: WillumpInputNode = WillumpInputNode("input_str")
        string_lower_node: StringLowerNode = \
            StringLowerNode(input_node=input_node, input_name="input_str",
                            input_type=WeldPandas([WeldStr()], ["target_col"]),
                            input_col="target_col",
                            output_name="lowered_output_words",
                            output_col="new_col",
                            output_type=WeldPandas([WeldVec(WeldStr()), WeldVec(WeldStr())],
                                                       ["new_col", "target_col"]))
        output_node: WillumpOutputNode = WillumpOutputNode(string_lower_node, ["lowered_output_words"])
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_str": WeldPandas([WeldVec(WeldStr())], ["target_col"]),
                    "lowered_output_words": WeldPandas([WeldVec(WeldStr()), WeldVec(WeldStr())],
                                                       ["new_col", "target_col"])}
        weld_output = wexec.execute_from_basics(graph, type_map, ((input_str,),), ["input_str"], ["lowered_output_words"],
                                                [])
        self.assertEqual(weld_output, (["aa,.,.a", "b,,b", "c34234c"], ["aA,.,.a", "B,,b", "c34234C"]))
