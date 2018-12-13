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


def graph_to_weld(graph: WillumpGraph) -> List[typing.Union[ast.AST, Tuple[str, List[str], str]]]:
    """
    Convert a Willump graph into a valid Weld program string.

    Graphs are converted by reversing their arrows (so that inputs are sources and outputs are
    sinks), topologically sorting them (so that all nodes come after all their inputs) and then
    concatenating the weld strings in the topological sort order.
    """
    # # Adjacency list representation of the reverse of the Willump graph (arrows point from nodes to their outputs).
    # reverse_graph: MutableMapping[WillumpGraphNode, List[WillumpGraphNode]] = {}
    # # Use DFS to build the forward and reverse graphs.
    # node_stack: List[WillumpGraphNode] = [graph.get_output_node()]
    # nodes_seen: Set[WillumpGraphNode] = set()
    # while len(node_stack) > 0:
    #     current_node: WillumpGraphNode = node_stack.pop()
    #     if current_node not in reverse_graph:
    #         reverse_graph[current_node] = []
    #     for node in current_node.get_in_nodes():
    #         if node not in reverse_graph:
    #             reverse_graph[node] = []
    #         reverse_graph[node].append(current_node)
    #         if node not in nodes_seen:
    #             nodes_seen.add(node)
    #             node_stack.append(node)
    sorted_nodes = topological_sort_graph(graph)
    # Construct the final list of Python calls and Weld blocks.  Each Weld block is a chunk of coalesced Weld code
    # that takes in some number of inputs and returns exactly one output.  No Weld node can appear in more than
    # one Weld block.  A Weld block is represented as a tuple (weld code,  weld input
    # variable names, weld output variable names).
    #
    # We construct Weld blocks from Weld sets, which are lists of Weld nodes in topo-sorted order that depend only
    # on already-executed Python code and may be dependencies for later-appearing Python code.  We construct a block
    # by starting with the last node in the set, defining its output to be the output of the block, and then recursively
    # building a Willump graph from its inputs and their inputs, stopping when an input either is not in our Weld set
    # or is a dependency of a node not in our Weld block.  Our Weld block is then the concatenated Weld code from the
    # topologically-sorted Weld set.
    # weld_python_list: List[typing.Union[ast.AST, Tuple[str, str, List[str]]]] = []
    # weld_start_index: int = 0
    # for i, node in enumerate(sorted_nodes):
    #     if isinstance(node, WillumpPythonNode):
    #         if (i - 1) != weld_start_index:
    #             weld_node_set = copy.copy(sorted_nodes[weld_start_index:i])
    #             while len(weld_node_set) > 0:
    #                 weld_block_root = copy.copy(weld_node_set.pop())
    #                 willump_output_node = WillumpOutputNode(weld_block_root)
    #                 weld_block_set: Set[WillumpGraphNode] = {willump_output_node}
    #                 weld_in_list: List[str] = []
    #                 weld_block_output: str = weld_block_root.get_output_name()
    #                 weld_block_stack = [weld_block_root]
    #                 while len(weld_block_stack) > 0:
    #                     curr_weld_node = weld_block_stack.pop()
    #                     weld_block_set.add(curr_weld_node)
    #                     if curr_weld_node not in weld_node_set or \
    #                             not set(reverse_graph[curr_weld_node]).issubset(weld_block_set):
    #                         weld_block_set.add(curr_weld_node)
    #                         weld_in_list.append(curr_variable_name)
    #
    #         weld_start_index = i
    #         weld_python_list.append(node.get_python())
    # return weld_python_list
    weld_python_list: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = []
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
            node_output_node: WillumpOutputNode = WillumpOutputNode(node)
            weld_str = ""
            for weld_node in node_input_list + [node] + [node_output_node]:
                weld_str += weld_node.get_node_weld()
            weld_python_list.append((weld_str, weld_in_list, node.get_output_name()))
    return weld_python_list
    # Concatenate the weld strings of all the topologically sorted nodes in order.
    # weld_str: str = ""
    # for node in sorted_nodes:
    #     weld_str += node.get_node_weld()
    # return weld_str


def set_input_names(weld_program: str, args_list: List[str], aux_data: List[Tuple[int, WeldType]]) \
        -> str:
    for i, arg in enumerate(args_list):
        weld_program = weld_program.replace("WELD_INPUT_{0}".format(arg), "_inp{0}".format(i))
    for i in range(len(aux_data)):
        weld_program = weld_program.replace("WELD_INPUT_AUX_DATA_{0}".format(i),
                                            "_inp{0}".format(len(args_list) + i))
    return weld_program
