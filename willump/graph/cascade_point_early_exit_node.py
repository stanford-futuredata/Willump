import ast

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var


class CascadePointEarlyExitNode(WillumpPythonNode):
    """
    Willump cascade point early exit node.  In a cascade in a point setting, check if the small model is confident in
    its output; if it is, return that output.
    """

    reshape_args: list

    def __init__(self, small_model_output_node: WillumpGraphNode, small_model_output_name: str):
        """
        Initialize the node.
        """
        input_nodes = [small_model_output_node]
        input_names = [small_model_output_name]
        python_ast = self.get_python_ast(small_model_output_name)
        super(CascadePointEarlyExitNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                          output_names=[], output_types=[],
                                          python_ast=python_ast)

    def get_python_ast(self, small_model_output_name) -> ast.AST:
        python_string = \
            """if {0}[0] != 2:\n""" \
            """\treturn {0}[0]""".format(strip_linenos_from_var(small_model_output_name))
        python_ast = ast.parse(python_string, "exec")
        return python_ast.body[0]

    def __repr__(self):
        return "Cascade Point Early Exit Node node for input {0}\n" \
            .format(self._input_names)
