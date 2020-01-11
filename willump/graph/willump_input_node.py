from willump.graph.willump_graph_node import WillumpGraphNode
from willump.willump_utilities import strip_linenos_from_var

from typing import List


class WillumpInputNode(WillumpGraphNode):
    """
    Willump graph input node.  Must be a sink of the graph.  Represents an input source.
    """
    _arg_name: str

    def __init__(self, arg_name: str, arg_type=None) -> None:
        self._arg_name = arg_name
        self._arg_type = arg_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return []

    def get_in_names(self) -> List[str]:
        return []

    def get_cost(self):
        return 0.0

    def get_node_weld(self) -> str:
        return "let {0} = WELD_INPUT_{0}__;".format(self._arg_name)

    def get_output_name(self) -> str:
        return self._arg_name

    def get_output_names(self) -> List[str]:
        return [self._arg_name]

    def get_output_types(self):
        return [self._arg_type]

    def __repr__(self):
        return "Input node for input {0}\n".format(self._arg_name)

    def graphviz_repr(self):
        return "Input: {0}".format(strip_linenos_from_var(self._arg_name))
