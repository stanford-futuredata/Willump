import ast

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var

from typing import List


class StackDenseNode(WillumpPythonNode):
    """
    Willump Stack Dense Node.  Horizontally stacks numpy arrays.
    """

    reshape_args: list

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], output_name: str,
                 output_type: WeldType):
        """
        Initialize the node.
        """
        assert (isinstance(output_type, WeldVec))
        self._output_name = output_name
        self._output_type = output_type
        python_ast = self.get_python_ast(input_names, output_name)
        super(StackDenseNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                             output_names=[output_name], output_types=[output_type],
                                             python_ast=python_ast)

    def get_python_ast(self, input_names, output_name) -> ast.AST:
        input_names_string = ""
        for name in input_names:
            input_names_string += "%s," % strip_linenos_from_var(name)
        python_string = "%s = np.hstack((%s))" % (strip_linenos_from_var(output_name),
                                                  input_names_string)
        python_ast = ast.parse(python_string, "exec")
        return python_ast.body[0]

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_type(self) -> WeldType:
        return self._output_type

    def __repr__(self):
        return "Stack dense node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
