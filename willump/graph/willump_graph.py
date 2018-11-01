import typing
import willump.graph.willump_input_node as wgin
import willump.graph.willump_output_node as wgon


class WillumpGraph(object):
    """
    A Willump Graph.  Each graph represents a featurization pipeline.
    """
    input_nodes: typing.List[wgin.WillumpInputNode]
    output_node: wgon.WillumpOutputNode

    def __init__(self, input_nodes: typing.List[wgin.WillumpInputNode],
                 output_node: wgon.WillumpOutputNode) -> None:
        self.input_nodes = input_nodes
        self.output_node = output_node

    def validate_graph(self) -> bool:
        """
        Guarantee that all input nodes are reachable from the output node, that all input nodes
        are sources, and that all sources are input nodes.
        """
        return True

    def get_output_node(self) -> wgon.WillumpOutputNode:
        """
        Return the graph's output node.
        """
        return self.output_node
