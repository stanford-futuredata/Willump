from typing import MutableMapping, List, Set, Tuple
import typing
import ast
import copy

from weld.types import *

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.willump_multioutput_node import WillumpMultiOutputNode


def topological_sort_graph(graph: WillumpGraph) -> List[WillumpGraphNode]:
    # Adjacency list representation of the Willump graph (arrows point from nodes to their inputs).
    forward_graph: MutableMapping[WillumpGraphNode, List[WillumpGraphNode]] = {}
    # Adjacency list representation of the reverse of the Willump graph (arrows point from nodes to their outputs).
    reverse_graph: MutableMapping[WillumpGraphNode, List[WillumpGraphNode]] = {}
    # Use DFS to build the forward and reverse graphs.
    node_stack: List[WillumpGraphNode] = [graph.get_output_node()]
    nodes_seen: Set[WillumpGraphNode] = set()
    while len(node_stack) > 0:
        current_node: WillumpGraphNode = node_stack.pop()
        forward_graph[current_node] = copy.copy(current_node.get_in_nodes())
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
    return sorted_nodes


def graph_to_weld(graph: WillumpGraph) -> List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]]:
    """
    Convert a Willump graph into a valid Weld program string.

    Graphs are converted by reversing their arrows (so that inputs are sources and outputs are
    sinks), topologically sorting them (so that all nodes come after all their inputs) and then
    concatenating the weld strings in the topological sort order.
    """
    sorted_nodes = topological_sort_graph(graph)
    weld_python_list: List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]] = []
    for node in sorted_nodes:
        if isinstance(node, WillumpPythonNode):
            weld_python_list.append(node.get_python())
        elif isinstance(node, WillumpInputNode) or isinstance(node, WillumpOutputNode):
            pass
        else:
            node_input_list: List[WillumpGraphNode] = []
            weld_in_list: List[str] = []
            for node_input in node.get_in_nodes():
                node_input_list.append(WillumpInputNode(node_input.get_output_name()))
                if "AUX_DATA" not in node_input.get_output_name():
                    weld_in_list.append(node_input.get_output_name())
            node_output_node: WillumpOutputNode = WillumpMultiOutputNode([node.get_output_name()])
            weld_str = ""
            for weld_node in node_input_list + [node] + [node_output_node]:
                weld_str += weld_node.get_node_weld()
            weld_python_list.append((weld_str, weld_in_list, [node.get_output_name()]))
    return weld_python_list


def set_input_names(weld_program: str, args_list: List[str], aux_data: List[Tuple[int, WeldType]]) \
        -> str:
    for i, arg in enumerate(args_list):
        weld_program = weld_program.replace("WELD_INPUT_{0}__".format(arg), "_inp{0}".format(i))
    for i in range(len(aux_data)):
        weld_program = weld_program.replace("WELD_INPUT_AUX_DATA_{0}__".format(i),
                                            "_inp{0}".format(len(args_list) + i))
    return weld_program
