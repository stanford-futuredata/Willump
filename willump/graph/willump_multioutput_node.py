from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class WillumpMultiOutputNode(WillumpGraphNode):
    """
    Not a standard graph node--has multiple outputs unlike all graph nodes.
    """
    input_list: List[str]

    def __init__(self, input_list: List[str]) -> None:
        self.input_list = input_list

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return []

    def get_node_type(self) -> str:
        return "multi-output"

    def get_node_weld(self) -> str:
        weld_str = "{"
        for entry in self.input_list:
            weld_str = weld_str + entry + ","
        weld_str += "}"
        return weld_str

    def get_output_name(self) -> str:
        return self.input_node.get_output_name()

    def __repr__(self):
        return "Multi-Output node\n"
