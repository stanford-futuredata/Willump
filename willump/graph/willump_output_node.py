import typing
import willump.graph.willump_graph_node as wgn


class WillumpOutputNode(wgn.WillumpGraphNode):
    """
    Willump graph output node.  Must be a source of the graph.  There can only be one output node,
    and it must be the graph's only source.  Represents the machine learning model the pipeline
    feeds into.
    """
    input_node: wgn.WillumpGraphNode

    def __init__(self, input_node: wgn.WillumpGraphNode) -> None:
        self.input_node = input_node

    def get_in_nodes(self) -> typing.List[wgn.WillumpGraphNode]:
        return [self.input_node]

    def get_node_type(self) -> str:
        return "output"

    def get_node_weld(self) -> str:
        return "{0}".format(self.input_node.get_output_name())

    def get_output_name(self) -> str:
        return ""

    def __repr__(self):
        return "Output node\n"
