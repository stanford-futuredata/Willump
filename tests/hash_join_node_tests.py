import unittest
import numpy
import pandas as pd
import sklearn.linear_model

import willump.evaluation.willump_executor as wexec

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.hash_join_node import WillumpHashJoinNode
from weld.types import *

model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)


@wexec.willump_execute()
def sample_pushing_merge_two_one_unused(left, right, right2):
    new1 = left.merge(right, how="left", on="join_column")
    new2 = new1.merge(right2, how="left", on="join_column")
    feature_columns = new2[FEATURES]
    predicted_result = model.predict(feature_columns)
    return predicted_result


@wexec.willump_execute()
def sample_pushing_merge_two(left, right, right2):
    new1 = left.merge(right, how="left", on="join_column")
    new2 = new1.merge(right2, how="left", on="join_column")
    feature_columns = new2[FEATURES]
    predicted_result = model.predict(feature_columns)
    return predicted_result


@wexec.willump_execute()
def sample_pushing_merge_three(left, right, right2, right3):
    new1 = left.merge(right, how="left", on="join_column")
    new2 = new1.merge(right2, how="left", on="join_column")
    new3 = new2.merge(right3, how="left", on="join_column")
    feature_columns = new3[FEATURES]
    predicted_result = model.predict(feature_columns)
    return predicted_result


@wexec.willump_execute()
def sample_merge_with_null(left, right):
    new = left.merge(right, how="left", on="join_column")
    new = new.fillna(0)
    return new


@wexec.willump_execute()
def sample_merge_with_pushed_null(left, right):
    new = left.merge(right, how="left", on="join_column")
    new = new.fillna(0)
    feature_columns = new[FEATURES]
    feature_columns = feature_columns.fillna(0)
    predicted_result = model.predict(feature_columns)
    return predicted_result


@wexec.willump_execute()
def sample_multicolumn_join(left, right, on):
    new = left.merge(right, how="left", on=on)
    return new


class HashJoinNodeTests(unittest.TestCase):
    def test_basic_hash_join(self):
        print("\ntest_basic_hash_join")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        input_node: WillumpInputNode = WillumpInputNode("input_table")
        aux_data = []
        hash_join_node: WillumpHashJoinNode = \
            WillumpHashJoinNode(input_node=input_node, input_name="input_table", output_name="output",
                                join_col_name="join_column",
                                right_dataframe=right_table, aux_data=aux_data,
                                left_input_type=WeldPandas(
                                    [WeldVec(WeldLong()), WeldVec(WeldLong()), WeldVec(WeldLong())],
                                    ["join_column", "data1", "data2"]))
        output_node: WillumpOutputNode = WillumpOutputNode(hash_join_node, ["output"])
        graph: WillumpGraph = WillumpGraph(output_node)
        type_map = {"input_table": WeldPandas([WeldVec(WeldLong()), WeldVec(WeldLong()), WeldVec(WeldLong())],
                                              ["join_column", "data1", "data2"]),
                    "output": WeldPandas([WeldVec(WeldLong()), WeldVec(WeldLong()), WeldVec(WeldLong()),
                                          WeldVec(WeldDouble()), WeldVec(WeldDouble())],
                                         ["join_column", "data1", "data2", "metadata1", "metadata2"])}
        weld_output = wexec.execute_from_basics(graph=graph,
                                                type_map=type_map,
                                                inputs=((left_table["join_column"].values,
                                                         left_table["data1"].values, left_table["data2"].values),),
                                                input_names=["input_table"],
                                                output_names=["output"],
                                                aux_data=aux_data)
        numpy.testing.assert_equal(
            weld_output[1], numpy.array([4, 5, 2, 5, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output[3], numpy.array([1.2, 2.2, 2.2, 3.2, 1.2], dtype=numpy.float64))
        numpy.testing.assert_equal(
            weld_output[4], numpy.array([1.3, 2.3, 2.3, 3.3, 1.3], dtype=numpy.float64))

    def test_pushing_merge_two_one_unused(self):
        print("\ntest_pushing_merge_two_one_unused")
        global FEATURES
        FEATURES = ["data1", "data2", "metadata1", "metadata2"]
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        right_table2 = pd.read_csv("tests/test_resources/toy_metadata_csv_2.csv")
        sample_pushing_merge_two_one_unused(left_table, right_table, right_table2)
        sample_pushing_merge_two_one_unused(left_table, right_table, right_table2)
        weld_output = sample_pushing_merge_two_one_unused(left_table, right_table, right_table2)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0, 1, 1, 1, 1]))

    def test_pushing_merge_two(self):
        print("\ntest_pushing_merge_two")
        global FEATURES
        FEATURES = ["data1", "data2", "metadata1", "metadata2", "metadata3", "metadata4"]
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        right_table2 = pd.read_csv("tests/test_resources/toy_metadata_csv_2.csv")
        correct_answer = sample_pushing_merge_two(left_table, right_table, right_table2)
        sample_pushing_merge_two(left_table, right_table, right_table2)
        weld_output = sample_pushing_merge_two(left_table, right_table, right_table2)
        numpy.testing.assert_equal(weld_output, correct_answer)

    def test_pushing_merge_three(self):
        print("\ntest_pushing_merge_three")
        global FEATURES
        FEATURES = ["data1", "data2", "metadata1", "metadata2", "metadata3", "metadata4", "metadata5", "metadata6"]
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        right_table2 = pd.read_csv("tests/test_resources/toy_metadata_csv_2.csv")
        right_table3 = pd.read_csv("tests/test_resources/toy_metadata_csv_3.csv")
        correct_answer = sample_pushing_merge_three(left_table, right_table, right_table2, right_table3)
        sample_pushing_merge_three(left_table, right_table, right_table2, right_table3)
        weld_output = sample_pushing_merge_three(left_table, right_table, right_table2, right_table3)
        numpy.testing.assert_equal(weld_output, correct_answer)

    def test_merge_with_null(self):
        print("\ntest_merge_with_null")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        left_table["join_column"] = [1, 2, 2, 5, 1]
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        correct_answer = sample_merge_with_null(left_table, right_table)
        sample_merge_with_null(left_table, right_table)
        weld_output = sample_merge_with_null(left_table, right_table)
        numpy.testing.assert_almost_equal(weld_output.values, correct_answer.values)

    def test_merge_with_null_pushing(self):
        print("\ntest_merge_with_null_pushing")
        global FEATURES
        FEATURES = ["data1", "data2", "metadata1", "metadata2"]
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        left_table["join_column"] = [1, 2, 2, 5, 1]
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        correct_output = sample_merge_with_pushed_null(left_table, right_table)
        sample_merge_with_pushed_null(left_table, right_table)
        weld_output = sample_merge_with_pushed_null(left_table, right_table)
        numpy.testing.assert_equal(
            weld_output, correct_output)

    def test_multicol_join(self):
        print("\ntest_multicol_join")
        on_cols = ["join_column", "join_column2"]
        left_table = pd.read_csv("tests/test_resources/toy_data_csv_multicol.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_multicol.csv")
        correct_output = sample_multicolumn_join(left_table, right_table, on_cols)
        sample_multicolumn_join(left_table, right_table, on_cols)
        weld_output = sample_multicolumn_join(left_table, right_table, on_cols)
        numpy.testing.assert_equal(
            weld_output.values, correct_output.values)

