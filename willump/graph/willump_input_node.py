from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class WillumpInputNode(WillumpGraphNode):
    """
    Willump graph input node.  Must be a sink of the graph.  Represents an input source.
    """
    _arg_name: str

    def __init__(self, arg_name: str) -> None:
        self._arg_name = arg_name

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return []

    def get_node_type(self) -> str:
        return "input"

    def get_node_weld(self) -> str:
        return "let {0} = WELD_INPUT_{0};".format(self._arg_name)

    def get_output_name(self) -> str:
        return self._arg_name

    def __repr__(self):
        return "Input node for input {0}\n".format(self._arg_name)
