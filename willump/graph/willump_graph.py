from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_output_node import WillumpOutputNode
from typing import List, Set


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
        ret_string: str = ""
        node_queue: List[WillumpGraphNode] = [self.get_output_node()]
        nodes_seen: Set[WillumpGraphNode] = set()
        while len(node_queue) > 0:
            current_node = node_queue.pop(0)
            ret_string += current_node.__repr__()
            for node in current_node.get_in_nodes():
                if node not in nodes_seen:
                    nodes_seen.add(node)
                    node_queue.append(node)
        return ret_string
