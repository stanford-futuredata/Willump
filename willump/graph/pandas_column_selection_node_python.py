import ast
from typing import List

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var


class PandasColumnSelectionNodePython(WillumpPythonNode):
    """
    Returns a dataframe containing a selection of columns from other dataframes.
    """
    selected_columns: List[str]

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], output_name: str,
                 input_types: List[WeldType], output_type: WeldType, selected_columns: List[str]) -> None:
        """
        Initialize the node.
        """
        assert(len(input_names) == 1)
        self._output_name = output_name
        self._input_nodes = input_nodes
        self._input_types = input_types
        self.selected_columns = selected_columns
        self._input_names = input_names
        self._output_type = output_type
        python_ast = self.get_python_ast(output_name, input_names[0], selected_columns)
        super(PandasColumnSelectionNodePython, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                          output_names=[output_name], output_types=[output_type],
                                          python_ast=python_ast)

    @staticmethod
    def get_python_ast(output_name, input_name, column_names) -> ast.AST:
        col_name_str = ""
        for col_name in column_names:
            col_name_str += "\"%s\"," % col_name
        python_string = "%s = %s[[%s]]" % \
                        (strip_linenos_from_var(output_name), strip_linenos_from_var(input_name), col_name_str)
        python_ast = ast.parse(python_string, "exec")
        return python_ast.body[0]

    def get_output_name(self):
        return self._output_name

    def get_output_type(self):
        return self._output_type

    def __repr__(self):
        return "Pandas Python Column Selection node for input {0} output {1}\n" \
            .format(self._input_names, self._output_names)
