import typing
import willump.graph.willump_graph_node as wgn


class WillumpInputNode(wgn.WillumpGraphNode):
    """
    Willump graph input node.  Must be a sink of the graph.  Represents an input source.
    """
    def get_in_nodes(self) -> typing.List[wgn.WillumpGraphNode]:
        return []

    def get_node_type(self) -> str:
        return "input"

    def get_node_weld(self) -> str:
        return ""
