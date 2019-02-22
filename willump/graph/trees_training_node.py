from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_training_node import WillumpTrainingNode

from typing import List
import ast
import numpy as np


class TreesTrainingNode(WillumpTrainingNode):
    """
    Trees Training Node.  Contains Python code to train a trees model.
    """

    def __init__(self, python_ast: ast.AST, input_names: List[str], output_names: List[str],
                 in_nodes: List[WillumpGraphNode], feature_importances,
                 is_async_node: bool = False) -> None:
        super(TreesTrainingNode, self).__init__(python_ast, input_names, output_names,
                                                in_nodes, feature_importances, is_async_node)

    def __repr__(self):
        return "Willump Trees Training Python node with inputs %s\n outputs %s\n and code %s\n" % \
               (str(list(map(lambda x: x.get_output_names(), self._in_nodes))), self._output_names,
                ast.dump(self._python_ast))
