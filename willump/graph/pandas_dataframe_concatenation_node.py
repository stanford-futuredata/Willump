import ast
from typing import List

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import strip_linenos_from_var


class PandasDataframeConcatenationNode(WillumpPythonNode):
    """
    Willump train-test split node.  Reshapes a numpy array.
    """

    keyword_args: list

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], input_types: List[WeldPandas],
                 output_name: str, output_type: WeldPandas, keyword_args) -> None:
        """
        Initialize the node.
        """
        python_ast = self.get_python_ast(output_name, input_names, keyword_args)
        self.keyword_args = keyword_args
        self.input_types = input_types
        self._output_name = output_name
        self._output_type = output_type
        super(PandasDataframeConcatenationNode, self).__init__(in_nodes=input_nodes, input_names=input_names,
                                          output_names=[output_name], output_types=[output_type],
                                          python_ast=python_ast)

    @staticmethod
    def get_python_ast(output_name, input_names, keyword_args) -> ast.AST:
        input_string = ""
        for input_name in input_names:
            input_string += "%s," % strip_linenos_from_var(input_name)
        python_string = "%s = pd.concat([%s])" % (strip_linenos_from_var(output_name), input_string)
        python_ast = ast.parse(python_string, "exec")
        python_ast.body[0].value.keywords = keyword_args
        return python_ast.body[0]

    def get_output_name(self):
        return self._output_name

    def get_output_type(self):
        return self._output_type

    def __repr__(self):
        return "Pandas Dataframe Cocnatenation node for input {0} output {1}\n" \
            .format(self._input_names, self._output_names)
