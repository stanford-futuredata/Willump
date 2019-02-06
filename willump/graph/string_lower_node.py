from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List

from weld.types import *


class StringLowerNode(WillumpGraphNode):
    """
    Willump String Lowering Node.  Takes in a list of strings and lowers all capital letters.
    Equivalent to mapping Python's lower() over a list of strings.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, input_type: WeldType, input_col: str,
                 output_name: str, output_type: WeldType, output_col: str) -> None:
        self._input_node = input_node
        self._input_df_name = input_name
        self._output_name = output_name
        self._input_names = [input_name]
        self._input_type = input_type
        self._input_col = input_col
        self._output_type = output_type
        self._output_col = output_col

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        assert(isinstance(self._input_type, WeldPandas))
        assert(isinstance(self._output_type, WeldPandas))
        input_target_col_num = self._input_type.column_names.index(self._input_col)
        output_statement = ""
        for output_column in self._output_type.column_names:
            if output_column != self._output_col:
                output_statement += "INPUT_NAME.$%d," % self._input_type.column_names.index(output_column)
            else:
                output_statement += "lowered_col,"
        weld_program = \
            """
            let lowered_col: vec[vec[i8]] = result(
                for(INPUT_NAME.$TARGET_COL,
                    appender[vec[i8]],
                    | bs: appender[vec[i8]], i: i64, n: vec[i8] |
                        let lowered_n = for(n,
                            appender[i8],
                            | ns: appender[i8], i_c: i64, char: i8 |
                                if(char >= 65c && char <= 90c, # If char is capital
                                    merge(ns, char + 32c), # Lower-case it
                                    merge(ns, char)
                                )  
                        );
                        merge(bs, result(lowered_n))
                )
            );
            let OUTPUT_NAME = {OUTPUT_STATEMENT};
            """
        weld_program = weld_program.replace("OUTPUT_STATEMENT", output_statement)
        weld_program = weld_program.replace("TARGET_COL", str(input_target_col_num))
        weld_program = weld_program.replace("INPUT_NAME", self._input_df_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "String Lower node for input {0} output {1}\n"\
            .format(self._input_df_name, self._output_name)
