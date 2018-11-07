from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode

import weld.weldobject
from weld.types import *
import weld.encoders

from typing import List, Set, MutableMapping

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()


def graph_to_weld(graph: WillumpGraph) -> str:
    """
    Convert a Willump graph into a valid Weld program string.

    Graphs are converted by reversing their arrows (so that inputs are sources and outputs are
    sinks), topologically sorting them (so that all nodes come after all their inputs) and then
    concatenating the weld strings in the topological sort order.
    """
    # Adjacency list representation of the Willump graph.
    forward_graph: MutableMapping[WillumpGraphNode, List[WillumpGraphNode]] = {}
    # Adjacency list representation of the reverse of the Willump graph.
    reverse_graph: MutableMapping[WillumpGraphNode, List[WillumpGraphNode]] = {}
    # Use DFS to build the forward and reverse graphs.
    node_stack: List[WillumpGraphNode] = [graph.get_output_node()]
    nodes_seen: Set[WillumpGraphNode] = set()
    while len(node_stack) > 0:
        current_node: WillumpGraphNode = node_stack.pop()
        forward_graph[current_node] = current_node.get_in_nodes()
        if current_node not in reverse_graph:
            reverse_graph[current_node] = []
        for node in current_node.get_in_nodes():
            if node not in reverse_graph:
                reverse_graph[node] = []
            reverse_graph[node].append(current_node)
            if node not in nodes_seen:
                nodes_seen.add(node)
                node_stack.append(node)
    # List of sources of reverse graph.
    reverse_graph_sources: List[WillumpGraphNode] = []
    for node, outputs in forward_graph.items():
        if len(outputs) == 0:
            reverse_graph_sources.append(node)
    # The nodes, topologically sorted.  Use Kahn's algorithm.
    sorted_nodes: List[WillumpGraphNode] = []
    while len(reverse_graph_sources) > 0:
        current_node = reverse_graph_sources.pop()
        sorted_nodes.append(current_node)
        while len(reverse_graph[current_node]) > 0:
            node = reverse_graph[current_node].pop()
            forward_graph[node].remove(current_node)
            if len(forward_graph[node]) == 0:
                reverse_graph_sources.append(node)
    # Concatenate the weld strings of all the topologically sorted nodes in order.
    weld_str: str = ""
    for node in sorted_nodes:
        weld_str += node.get_node_weld()
    return weld_str


def evaluate_weld(weld_program: str, input_vector):
    """
    Evaluate a Weld program string on a Numpy input vector.

    TODO:  Support multiple input vectors.

    TODO:  Support inputs which are not numpy arrays of type float64.
    """
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    vec_name = weld_object.update(input_vector, tys=WeldVec(WeldDouble()))
    weld_object.weld_code = weld_program.format(vec_name)
    return weld_object.evaluate(WeldVec(WeldDouble()), verbose=False)



