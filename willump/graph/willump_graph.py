from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_output_node import WillumpOutputNode
from typing import List


class WillumpGraph(object):
    """
    A Willump Graph.  Each graph represents a featurization pipeline.
    """
    output_node: WillumpOutputNode

    def __init__(self,
                 output_node: WillumpOutputNode) -> None:
        self.output_node = output_node

    def validate_graph(self) -> bool:
        """
        Guarantee that all input nodes are reachable from the output node, that all input nodes
        are sources, and that all sources are input nodes.
        """
        return True

    def get_output_node(self) -> WillumpOutputNode:
        """
        Return the graph's output node.
        """
        return self.output_node

    def __repr__(self) -> str:
        node_queue: List[WillumpGraphNode] = [self.output_node]
        ret_string: str = ""
        while len(node_queue) > 0:
            curr_node: WillumpGraphNode = node_queue[0]
            ret_string += curr_node.__repr__()
            node_queue = node_queue[1:] + curr_node.get_in_nodes()
        return ret_string
