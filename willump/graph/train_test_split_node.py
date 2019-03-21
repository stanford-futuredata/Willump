import ast

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var


class TrainTestSplitNode(WillumpPythonNode):
    """
    Willump train-test split node.  Reshapes a numpy array.
    """

    reshape_args: list

    def __init__(self, input_node: WillumpGraphNode, input_name: str, train_name: str, test_name: str,
                 output_type: WeldType, keyword_args: list):
        """
        Initialize the node.
        """
        input_nodes = [input_node]
        input_names = [input_name]
        python_ast = self.get_python_ast(input_name, train_name, test_name, keyword_args)
        self.keyword_args = keyword_args
        super(TrainTestSplitNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                          output_names=[train_name, test_name], output_types=[output_type, output_type],
                                          python_ast=python_ast)

    def get_python_ast(self, input_name, train_name, test_name, keyword_args) -> ast.AST:
        python_string = "%s, %s = train_test_split(%s)" % (strip_linenos_from_var(train_name),
                                                    strip_linenos_from_var(test_name),
                                                           strip_linenos_from_var(input_name))
        python_ast = ast.parse(python_string, "exec")
        python_ast.body[0].value.keywords = keyword_args
        return python_ast.body[0]

    def __repr__(self):
        return "Train-test split node for input {0} output {1}\n" \
            .format(self._input_names, self._output_names)
