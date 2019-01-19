from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List


class PandasColumnSelectionNode(WillumpGraphNode):
    """
    Returns a dataframe containing a selection of columns from another dataframe.
    """
    selected_columns: List[str]

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_type: WeldType, selected_columns: List[str]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name
        self._input_nodes = [input_node]
        self._input_type = input_type
        self.selected_columns = selected_columns

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

    def get_node_weld(self) -> str:
        assert(isinstance(self._input_type, WeldPandas))
        selection_string = ""
        input_columns = self._input_type.column_names
        for column in self.selected_columns:
            selection_string += "INPUT_NAME.$%d," % input_columns.index(column)
        weld_program = "let OUTPUT_NAME = {%s};" % selection_string
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Pandas column selection node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
