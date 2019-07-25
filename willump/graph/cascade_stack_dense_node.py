from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode

from weld.types import *

import ast
from typing import List
from willump.willump_utilities import strip_linenos_from_var


class CascadeStackDenseNode(WillumpPythonNode):
    """
    Willump Stack Dense node.  Horizontally stacks multiple dense matrices.
    """

    def __init__(self, more_important_nodes: List[WillumpGraphNode], more_important_names: List[str],
                 less_important_nodes: List[WillumpGraphNode], less_important_names: List[str],
                 output_name: str, output_type: WeldType, small_model_output_node: WillumpGraphNode,
                 small_model_output_name: str) -> None:
        """
        Initialize the node.
        """
        assert (isinstance(output_type, WeldVec))
        assert (isinstance(output_type.elemType, WeldVec))
        self._output_name = output_name
        self._output_type = output_type
        input_nodes = more_important_nodes + less_important_nodes + [small_model_output_node]
        input_names = more_important_names + less_important_names + [small_model_output_name]
        python_ast = self.get_python_ast(more_important_names, less_important_names, small_model_output_name,
                                         output_name)
        super(CascadeStackDenseNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                              output_names=[output_name], output_types=[output_type],
                                              python_ast=python_ast)

    def get_python_ast(self, more_important_names, less_important_names, small_model_output_name,
                       output_name) -> ast.AST:
        more_important_vecs = [strip_linenos_from_var(name) for name in more_important_names]
        less_important_vecs = [strip_linenos_from_var(name) for name in less_important_names]
        more_important_str, less_important_str = "", ""
        for vec in more_important_vecs:
            more_important_str += "%s," % vec
        for vec in less_important_vecs:
            less_important_str += "%s," % vec
        python_string = "%s = cascade_dense_stacker([%s], [%s], %s)" % (
            strip_linenos_from_var(output_name), more_important_str,
            less_important_str, strip_linenos_from_var(small_model_output_name))
        python_ast = ast.parse(python_string, "exec")
        return python_ast.body[0]

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_type(self) -> WeldType:
        return self._output_type

    def __repr__(self):
        return "Stack dense node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
