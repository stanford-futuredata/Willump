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
    """
    _input_nodes: List[WillumpGraphNode]

    def __init__(self, input_node: WillumpGraphNode, output_name: str, join_col_name: str,
                 right_dataframe, aux_data: List[Tuple[int, WeldType]]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name
        self._right_dataframe_size = len(right_dataframe)
        self._right_dataframe_name = "AUX_DATA_{0}".format(len(aux_data))
        self._join_col_name = join_col_name
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._right_dataframe_name))
        for entry in self._process_aux_data(right_dataframe):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

    def _process_aux_data(self, right_dataframe) -> List[Tuple[int, WeldType]]:
        """
        Returns a pointer to a Weld dict[i64, {vec...vec}] of all the columns in right_dataframe indexed by join_col_name.
        """
        join_col_index = list(right_dataframe.columns).index(self._join_col_name)

        types_string = "{"
        for column in right_dataframe:
            if column != self._join_col_name:
                types_string += str(numpy_type_to_weld_type(right_dataframe[column].values.dtype)) + ","
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
        return [(indexed_right_dataframe, WeldDict(WeldLong(), WeldStruct([WeldDouble()])))]

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let list_dicts: vec[dict[i64, i64]] = result(for(INPUT_NAME,
                appender[dict[i64, i64]],
                | count_dicts: appender[dict[i64, i64]], i_out: i64, string: vec[i8] |
                    let string_len: i64 = len(string);
                    let string_dict: dict[i64, i64] = result(for(string,
                        dictmerger[i64, i64, +],
                        | count_dict:  dictmerger[i64, i64, +], i_string: i64, char: i8 |
                        for(rangeiter(NGRAM_MINL, NGRAM_MAXL, 1L),
                            count_dict,
                            | count_dict_inner: dictmerger[i64, i64, +], num_iter, iter_value |
                                if(i_string + iter_value <= string_len,
                                    let word: vec[i8] = slice(string, i_string, iter_value);
                                    let exists_and_key = optlookup(VOCAB_DICT_NAME, word);
                                    if(exists_and_key.$0,
                                        merge(count_dict_inner, {exists_and_key.$1, 1L}),
                                        count_dict_inner
                                    ),
                                    count_dict_inner    
                                )
                        )
                    ));
                    merge(count_dicts, string_dict)
            ));
            let vec_dict_vecs: vec[vec[{i64, i64}]] = map(list_dicts, 
                | string_dict: dict[i64, i64] | 
                    tovec(string_dict)
            );
            let out_struct: {appender[i64], appender[i64], appender[i64]} = for(vec_dict_vecs,
                {appender[i64], appender[i64], appender[i64]}, # Row number, index number, frequency
                | bs: {appender[i64], appender[i64], appender[i64]}, i: i64, x: vec[{i64, i64}] |
                    for(x,
                        bs,
                        | bs2: {appender[i64], appender[i64], appender[i64]}, i2: i64, x2: {i64, i64} |
                            {merge(bs2.$0, i), merge(bs2.$1, x2.$0), merge(bs2.$2, x2.$1)}
                    )
            );
            let ros: {vec[i64], vec[i64], vec[i64]} =
                {result(out_struct.$0), result(out_struct.$1), result(out_struct.$2)};

            let OUTPUT_NAME: vec[vec[i64]] = [ros.$0, ros.$1, ros.$2];
            """
        weld_program = weld_program.replace("VOCAB_DICT_NAME", self._vocab_dict_name)
        weld_program = weld_program.replace("BIG_ZEROS_VECTOR", self._big_zeros_vector_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("NGRAM_MIN", str(self._min_gram))
        weld_program = weld_program.replace("NGRAM_MAX", str(self._max_gram))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
