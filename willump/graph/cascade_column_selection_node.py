from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List


class CascadeColumnSelectionNode(WillumpGraphNode):
    """
    Cascade column selection node.  Selects columns from input dataframes, trimming the ones from more important
    input dataframes to match the length of those from the less important dataframes.
    """

    def __init__(self, more_important_nodes: List[WillumpGraphNode], more_important_names: List[str],
                 more_important_types: List[WeldType],
                 less_important_nodes: List[WillumpGraphNode], less_important_names: List[str],
                 less_important_types: List[WeldType],
                 output_name: str, small_model_output_node: WillumpGraphNode,
                 small_model_output_name: str,
                 selected_columns: List[str]) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = more_important_nodes + less_important_nodes + [small_model_output_node]
        self._output_name = output_name
        self._input_names = more_important_names + less_important_names + [small_model_output_name]
        self._more_important_names = more_important_names
        self._less_important_names = less_important_names
        self._small_model_output_name = small_model_output_name
        self._more_important_types = more_important_types
        self._less_important_types = less_important_types
        self._selected_columns = selected_columns

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        assert(all(isinstance(input_type, WeldPandas) for input_type in self._more_important_types))
        assert(all(isinstance(input_type, WeldPandas) for input_type in self._less_important_types))
        appender_list = ""
        merge_statement = ""
        col_to_preoutput = {}
        i = 0
        for column in self._selected_columns:
            for input_name, input_type in zip(self._more_important_names, self._more_important_types):
                input_columns = input_type.column_names
                if column in input_columns:
                    selected_column_index = input_columns.index(column)
                    appender_list += "appender[%s](output_len)," % (str(input_type.field_types[selected_column_index].elemType))
                    merge_statement += "merge(bs.$%d, lookup(%s.$%d, i))," % (i, input_name, selected_column_index)
                    col_to_preoutput[column] = i
                    i += 1
                    break
        combine_statement = ""
        for column in self._selected_columns:
            if column in col_to_preoutput:
                col_index = col_to_preoutput[column]
                combine_statement += "result(pre_output.$%d)," % col_index
            else:
                for input_name, input_type in zip(self._less_important_names, self._less_important_types):
                    input_columns = input_type.column_names
                    if column in input_columns:
                        selected_column_index = input_columns.index(column)
                        combine_statement += "%s.$%d," % (input_name, selected_column_index)
                        break
        weld_program = \
            """
            let output_len = len(SMALL_OUTPUT_NAME);
            let pre_output = for(SMALL_OUTPUT_NAME,
                {APPENDER_LIST},
                | bs, i: i64, x: i8 |
                if(x==2c,
                    {MERGE_STATEMENT},
                    bs
                )
            );
            let OUTPUT_NAME = {COMBINE_STATEMENT};
            """
        weld_program = weld_program.replace("SMALL_OUTPUT_NAME", self._small_model_output_name)
        weld_program = weld_program.replace("APPENDER_LIST", appender_list)
        weld_program = weld_program.replace("MERGE_STATEMENT", merge_statement)
        weld_program = weld_program.replace("COMBINE_STATEMENT", combine_statement)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Cascade column selection node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
