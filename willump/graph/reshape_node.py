import ast

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var


class ReshapeNode(WillumpPythonNode):
    """
    Willump Stack Dense node.  Horizontally stacks multiple dense matrices.
    """

    reshape_args: list

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str, output_type: WeldType,
                 reshape_args: list):
        """
        Initialize the node.
        """
        assert (isinstance(output_type, WeldVec))
        self._output_name = output_name
        self._output_type = output_type
        input_nodes = [input_node]
        input_names = [input_name]
        self.reshape_args = reshape_args
        python_ast = self.get_python_ast(input_name, output_name, reshape_args)
        super(ReshapeNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                          output_names=[output_name], output_types=[output_type],
                                          python_ast=python_ast)

    def get_python_ast(self, input_name, output_name, reshape_args) -> ast.AST:
        python_string = "%s = %s.reshape(1, -1)" % (strip_linenos_from_var(output_name),
                                                    strip_linenos_from_var(input_name))
        python_ast = ast.parse(python_string, "exec")
        python_ast.body[0].value.args = reshape_args
        return python_ast.body[0]

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_type(self) -> WeldType:
        return self._output_type

    def __repr__(self):
        return "Reshape node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
