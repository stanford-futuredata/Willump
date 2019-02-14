from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.willump_utilities import *

from weld.types import *

import pandas as pd
from typing import List, Tuple, Mapping, Optional
import importlib


class WillumpHashJoinNode(WillumpGraphNode):
    """
    Implements a left join between two dataframes on a specific column.

    TODO:  Handle case where entry is in left but not in right.
    """
    left_input_name: str
    left_df_type: WeldPandas
    right_df_row_type: WeldPandas
    right_df_type: WeldPandas
    join_col_name: str
    batch: bool
    output_type: WeldType

    # Protected Cascade Variables
    _small_model_output_name: Optional[str] = None

    def __init__(self, input_node: WillumpGraphNode, input_name: str, left_input_type: WeldType, output_name: str, join_col_name: str,
                 right_dataframe, aux_data: List[Tuple[int, WeldType]], batch=True) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self.left_input_name = input_name
        self._output_name = output_name
        self._right_dataframe = right_dataframe
        self._right_dataframe_name = "AUX_DATA_{0}".format(len(aux_data))
        self.join_col_name = join_col_name
        self._input_nodes = [input_node, WillumpInputNode(self._right_dataframe_name)]
        self._input_names = [input_name, self._right_dataframe_name]
        assert(isinstance(left_input_type, WeldPandas))
        self.left_df_type = left_input_type
        self._join_col_elem_type = left_input_type.field_types[left_input_type.column_names.index(join_col_name)]
        if isinstance(self._join_col_elem_type, WeldVec):
            self._join_col_elem_type = self._join_col_elem_type.elemType
        self.batch = batch
        self._model_type = None
        self._model_parameters = None
        self._model_index_map = None
        for entry in self._process_aux_data(right_dataframe):
            aux_data.append(entry)
        self.output_type = WeldPandas(field_types=self.left_df_type.field_types + self.right_df_type.field_types,
                                      column_names=self.left_df_type.column_names + self.right_df_type.column_names)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def _process_aux_data(self, right_dataframe) -> List[Tuple[int, WeldType]]:
        """
        Returns a pointer to a Weld dict[join_col_type, {vec...vec}] of all the columns in right_dataframe
        indexed by join_col_name.
        """
        import willump.evaluation.willump_executor as wexec
        join_col_index = list(right_dataframe.columns).index(self.join_col_name)

        types_string = "{"
        row_types_list = []
        types_list = []
        right_df_column_names = []
        for column in right_dataframe:
            if column != self.join_col_name:
                col_weld_type: WeldType = numpy_type_to_weld_type(right_dataframe[column].values.dtype)
                types_string += str(col_weld_type) + ","
                row_types_list.append(col_weld_type)
                if self.batch:
                    types_list.append(WeldVec(col_weld_type))
                else:
                    types_list.append(col_weld_type)
                right_df_column_names.append(column)
        types_string = types_string[:-1] + "}"

        values_string = "{"
        for i, column in enumerate(right_dataframe):
            if column != self.join_col_name:
                values_string += "lookup(_inp%d, i)," % i
        values_string = values_string[:-1] + "}"

        weld_program = \
            """
            result(for(_inp%d,
                dictmerger[JOIN_COL_TYPE, %s, +],
                | bs: dictmerger[JOIN_COL_TYPE, %s, +], i: i64, x |
                    merge(bs, {JOIN_COL_TYPE(x), %s})
            ))
            """ % (join_col_index, types_string, types_string, values_string)
        weld_program = weld_program.replace("JOIN_COL_TYPE", str(self._join_col_elem_type))

        input_arg_types = {}
        input_arg_list = []
        for i, column in enumerate(right_dataframe):
            input_name = "input%d" % i
            input_arg_types[input_name] = \
                WeldVec(numpy_type_to_weld_type(right_dataframe[column].values.dtype))
            input_arg_list.append(input_name)

        module_name = wexec.compile_weld_program(weld_program,
                                                 input_arg_types,
                                                 input_names=input_arg_list,
                                                 output_names=[],
                                                 base_filename="hash_join_dataframe_indexer")

        hash_join_dataframe_indexer = importlib.import_module(module_name)
        input_args = []
        for column in right_dataframe:
            input_args.append(right_dataframe[column].values)
        input_args = tuple(input_args)

        indexed_right_dataframe = hash_join_dataframe_indexer.caller_func(*input_args)
        self.right_df_row_type = WeldPandas(row_types_list, right_df_column_names)
        self.right_df_type = WeldPandas(types_list, right_df_column_names)
        return [(indexed_right_dataframe, WeldDict(self._join_col_elem_type, self.right_df_row_type))]

    def push_model(self, model_type: str, model_parameters: Tuple, model_index_map: Mapping[str, int]):
        self._model_type = model_type
        self._model_parameters = model_parameters
        self._model_index_map = model_index_map

    def push_cascade(self, small_model_output_name: str):
        self._small_model_output_name = small_model_output_name

    def get_node_weld(self) -> str:
        join_col_left_index = self.left_df_type.column_names.index(self.join_col_name)
        if self._model_type is None:
            if self._small_model_output_name is None:
                cascade_statement = "true"
            else:
                cascade_statement = "lookup(%s, i) == 2c" % self._small_model_output_name
            if self.batch:
                struct_builder_statement = "{"
                merge_statement = "{"
                result_statement = "{"
                switch = 0
                for i in range(len(self.left_df_type.column_names)):
                    result_statement += "%s.$%d," % (self.left_input_name, i)
                for i, column in enumerate(self._right_dataframe):
                    if column != self.join_col_name:
                        col_type = str(numpy_type_to_weld_type(self._right_dataframe[column].values.dtype))
                        struct_builder_statement += "appender[%s](col_len)," % col_type
                        merge_statement += "merge(bs.$%d, right_dataframe_row.$%d)," % (i - switch, i - switch)
                        result_statement += "result(pre_output.$%d)," % (i - switch)
                    else:
                        switch = 1
                struct_builder_statement = struct_builder_statement[:-1] + "}"
                merge_statement = merge_statement[:-1] + "}"
                result_statement = result_statement[:-1] + "}"
                weld_program = \
                    """
                    let col_len = len(INPUT_NAME.$JOIN_COL_LEFT_INDEX);
                    let pre_output = (for(INPUT_NAME.$JOIN_COL_LEFT_INDEX,
                        STRUCT_BUILDER,
                        |bs, i: i64, x |
                            if(CASCADE_STATEMENT,
                                let right_dataframe_row = lookup(RIGHT_DATAFRAME_NAME, x);
                                MERGE_STATEMENT,
                                bs
                            )
                    ));
                    let OUTPUT_NAME = RESULT_STATEMENT;
                    """
                weld_program = weld_program.replace("JOIN_COL_LEFT_INDEX", str(join_col_left_index))
                weld_program = weld_program.replace("STRUCT_BUILDER", struct_builder_statement)
                weld_program = weld_program.replace("MERGE_STATEMENT", merge_statement)
                weld_program = weld_program.replace("RESULT_STATEMENT", result_statement)
                weld_program = weld_program.replace("RIGHT_DATAFRAME_NAME", self._right_dataframe_name)
                weld_program = weld_program.replace("INPUT_NAME", self.left_input_name)
                weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
                weld_program = weld_program.replace("CASCADE_STATEMENT", cascade_statement)
            else:
                result_statement = "{"
                switch = 0
                for i in range(len(self.left_df_type.column_names)):
                    result_statement += "%s.$%d," % (self.left_input_name, i)
                for i, column in enumerate(self._right_dataframe):
                    if column != self.join_col_name:
                        result_statement += "pre_output.$%d," % (i - switch)
                    else:
                        switch = 1
                result_statement += "}"
                weld_program = \
                    """
                        let pre_output = lookup(RIGHT_DATAFRAME_NAME, INPUT_NAME.$JOIN_COL_LEFT_INDEX);
                        let OUTPUT_NAME = RESULT_STATEMENT;
                    """
                weld_program = weld_program.replace("JOIN_COL_LEFT_INDEX", str(join_col_left_index))
                weld_program = weld_program.replace("RIGHT_DATAFRAME_NAME", self._right_dataframe_name)
                weld_program = weld_program.replace("RESULT_STATEMENT", result_statement)
                weld_program = weld_program.replace("INPUT_NAME", self.left_input_name)
                weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        else:
            assert(self._model_type == 'linear')
            weights_name, = self._model_parameters
            left_index_map = {}
            for i, col_name in enumerate(self.left_df_type.column_names):
                if col_name in self._model_index_map:
                    left_index_map[i] = self._model_index_map[col_name]
            right_index_map = {}
            for i, col_name in enumerate(self.right_df_row_type.column_names):
                if col_name in self._model_index_map:
                    right_index_map[i] = self._model_index_map[col_name]
            if self.batch:
                sum_statement = ""
                for left_index, weight_index in left_index_map.items():
                    sum_statement += "f64(lookup(INPUT_NAME.$%d, i)) * lookup(WEIGHTS_NAME, %dL)+" % (left_index, weight_index)
                for right_index, weight_index in right_index_map.items():
                    sum_statement += "f64(right_dataframe_row.$%d) * lookup(WEIGHTS_NAME, %dL)+" % (right_index, weight_index)
                sum_statement = sum_statement[:-1]
                weld_program = \
                    """
                    let OUTPUT_NAME: vec[f64] = result(for(INPUT_NAME.$JOIN_COL_LEFT_INDEX,
                        appender[f64],
                        |bs, i: i64, x |
                            let right_dataframe_row = lookup(RIGHT_DATAFRAME_NAME, x);
                            merge(bs, SUM_STATEMENT)
                    ));
                    """
            else:
                sum_statement = ""
                for left_index, weight_index in left_index_map.items():
                    sum_statement += "f64(INPUT_NAME.$%d) * lookup(WEIGHTS_NAME, %dL)+" % (left_index, weight_index)
                for right_index, weight_index in right_index_map.items():
                    sum_statement += "f64(right_dataframe_row.$%d) * lookup(WEIGHTS_NAME, %dL)+" % (right_index, weight_index)
                sum_statement = sum_statement[:-1]
                weld_program = \
                    """
                    let right_dataframe_row = lookup(RIGHT_DATAFRAME_NAME, INPUT_NAME.$JOIN_COL_LEFT_INDEX);
                    let pre_output: f64 = SUM_STATEMENT;
                    let OUTPUT_NAME: vec[f64] = result(merge(appender[f64], pre_output));
                    """
            weld_program = weld_program.replace("SUM_STATEMENT", sum_statement)
            weld_program = weld_program.replace("JOIN_COL_LEFT_INDEX", str(join_col_left_index))
            weld_program = weld_program.replace("RIGHT_DATAFRAME_NAME", self._right_dataframe_name)
            weld_program = weld_program.replace("INPUT_NAME", self.left_input_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
            weld_program = weld_program.replace("WEIGHTS_NAME", weights_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Hash-join node for input {0} output {1}\n" \
            .format(self.left_input_name, self._output_name)
