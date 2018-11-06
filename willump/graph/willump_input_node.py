import typing
import willump.graph.willump_graph_node as wgn


class WillumpInputNode(wgn.WillumpGraphNode):
    """
    Willump graph input node.  Must be a sink of the graph.  Represents an input source.
    """
    _arg_name: str

    def __init__(self, arg_name: str) -> None:
        self._arg_name = arg_name

    def get_in_nodes(self) -> typing.List[wgn.WillumpGraphNode]:
        return []

    def get_node_type(self) -> str:
        return "input"

    def get_node_weld(self) -> str:
        return ""

    def get_arg_name(self) -> str:
        return self._arg_name

    def __repr__(self):
        return "Input node for input {0}\n".format(self._arg_name)
