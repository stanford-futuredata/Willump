from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class WillumpOutputNode(WillumpGraphNode):
    """
    Willump graph output node.  Must be a source of the graph.  There can only be one output node,
    and it must be the graph's only source.  Represents the machine learning model the pipeline
    feeds into.
    """
    _input_node: WillumpGraphNode

    def __init__(self, input_node: WillumpGraphNode, input_names: List[str]) -> None:
        self._input_node = input_node
        self._input_names = input_names

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        return "{0}\n".format(self._input_node.get_output_name())

    def get_output_name(self) -> str:
        return self._input_node.get_output_name()

    def __repr__(self):
        return "Output node\n"
