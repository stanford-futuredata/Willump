from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
import willump.evaluation.willump_executor as wexec
from willump.willump_utilities import *

from weld.types import *

import pandas as pd
from typing import List, Tuple, Mapping
import importlib


class WillumpHashJoinNode(WillumpGraphNode):
    """
    Implements an inner join between two dataframes on a specific column, preserving the order of the keys in the
    left dataframe.

    TODO:  Proper typing of join column instead of casting to i64.
    """
    _input_nodes: List[WillumpGraphNode]

    def __init__(self, input_node: WillumpGraphNode, output_name: str, join_col_name: str,
                 right_dataframe, aux_data: List[Tuple[int, WeldType]], size_left_df: int,
                 join_col_left_index: int, batch=True) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name
        self._right_dataframe = right_dataframe
        self._right_dataframe_name = "AUX_DATA_{0}".format(len(aux_data))
        self._join_col_name = join_col_name
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._right_dataframe_name))
        self._size_left_df = size_left_df
        self._join_col_left_index = join_col_left_index
        self.batch = batch
        for entry in self._process_aux_data(right_dataframe):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

    def _process_aux_data(self, right_dataframe) -> List[Tuple[int, WeldType]]:
        """
        Returns a pointer to a Weld dict[i64, {vec...vec}] of all the columns in right_dataframe
        indexed by join_col_name.
        """
        join_col_index = list(right_dataframe.columns).index(self._join_col_name)

        types_string = "{"
        types_list = []
        for column in right_dataframe:
            if column != self._join_col_name:
                col_weld_type: WeldType = numpy_type_to_weld_type(right_dataframe[column].values.dtype)
                types_string += str(col_weld_type) + ","
                types_list.append(col_weld_type)
        types_string = types_string[:-1] + "}"

        values_string = "{"
        for i, column in enumerate(right_dataframe):
            if column != self._join_col_name:
                values_string += "lookup(_inp%d, i)," % i
        values_string = values_string[:-1] + "}"

        weld_program = \
            """
            result(for(_inp%d,
                dictmerger[i64, %s, +],
                | bs: dictmerger[i64, %s, +], i: i64, x: i64 |
                    merge(bs, {x, %s})
            ))
            """ % (join_col_index, types_string, types_string, values_string)

        input_arg_types = {}
        for i, column in enumerate(right_dataframe):
            input_arg_types["__willump_arg%d" % i] = \
                WeldVec(numpy_type_to_weld_type(right_dataframe[column].values.dtype))

        module_name = wexec.compile_weld_program(weld_program,
                                                 input_arg_types,
                                                 base_filename="hash_join_dataframe_indexer")

        hash_join_dataframe_indexer = importlib.import_module(module_name)
        input_args = []
        for column in right_dataframe:
            input_args.append(right_dataframe[column].values)
        input_args = tuple(input_args)

        indexed_right_dataframe = hash_join_dataframe_indexer.caller_func(*input_args)
        return [(indexed_right_dataframe, WeldDict(WeldLong(), WeldPandas(types_list)))]

    def get_node_weld(self) -> str:
        if self.batch:
            struct_builder_statement = "{"
            merge_statement = "{"
            result_statement = "{"
            switch = 0
            for i in range(self._size_left_df):
                result_statement += "%s.$%d," % (self._input_array_string_name, i)
            for i, column in enumerate(self._right_dataframe):
                if column != self._join_col_name:
                    col_type = str(numpy_type_to_weld_type(self._right_dataframe[column].values.dtype))
                    struct_builder_statement += "appender[%s]," % col_type
                    merge_statement += "merge(bs.$%d, right_dataframe_row.$%d)," % (i - switch, i - switch)
                    result_statement += "result(pre_output.$%d)," % (i - switch)
                else:
                    switch = 1
            struct_builder_statement = struct_builder_statement[:-1] + "}"
            merge_statement = merge_statement[:-1] + "}"
            result_statement = result_statement[:-1] + "}"
            weld_program = \
                """
                let pre_output = (for(INPUT_NAME.$JOIN_COL_LEFT_INDEX,
                    STRUCT_BUILDER,
                    |bs, i: i64, x |
                        let right_dataframe_row = lookup(RIGHT_DATAFRAME_NAME, i64(x));
                        MERGE_STATEMENT
                ));
                let OUTPUT_NAME = RESULT_STATEMENT;
                """
            weld_program = weld_program.replace("JOIN_COL_LEFT_INDEX", str(self._join_col_left_index))
            weld_program = weld_program.replace("STRUCT_BUILDER", struct_builder_statement)
            weld_program = weld_program.replace("MERGE_STATEMENT", merge_statement)
            weld_program = weld_program.replace("RESULT_STATEMENT", result_statement)
            weld_program = weld_program.replace("RIGHT_DATAFRAME_NAME", self._right_dataframe_name)
            weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        else:
            result_statement = "{"
            switch = 0
            for i in range(self._size_left_df):
                result_statement += "%s.$%d," % (self._input_array_string_name, i)
            for i, column in enumerate(self._right_dataframe):
                if column != self._join_col_name:
                    result_statement += "pre_output.$%d," % (i - switch)
                else:
                    switch = 1
            result_statement += "}"
            weld_program = \
                """
                    let pre_output = lookup(RIGHT_DATAFRAME_NAME, i64(INPUT_NAME.$JOIN_COL_LEFT_INDEX));
                    let OUTPUT_NAME = RESULT_STATEMENT;
                """
            weld_program = weld_program.replace("JOIN_COL_LEFT_INDEX", str(self._join_col_left_index))
            weld_program = weld_program.replace("RIGHT_DATAFRAME_NAME", self._right_dataframe_name)
            weld_program = weld_program.replace("RESULT_STATEMENT", result_statement)
            weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
