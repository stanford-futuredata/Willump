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
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.combine_linear_regression_node import CombineLinearRegressionNode


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


def pushing_model_pass(weld_block_node_list, weld_block_output_set) -> List[WillumpGraphNode]:
    """
    Optimization pass that pushes a linear model into its input nodes in a block when possible.  Assumes block contains
    at most one model node, which must be the last node in the block.

    # TODO:  Support more classes of models.
    # TODO:  Support pushing through joins, etc.
    """
    model_node = weld_block_node_list[-1]
    if not isinstance(model_node, LinearRegressionNode):
        return weld_block_node_list
    input_node = model_node.get_in_nodes()[0]
    if isinstance(input_node, ArrayCountVectorizerNode) \
            and input_node in weld_block_node_list \
            and input_node.get_output_name() not in weld_block_output_set:
        input_node.push_model("linear", (model_node.weights_data_name,))
        weld_block_node_list.remove(model_node)
        weld_block_node_list.append(CombineLinearRegressionNode([input_node], model_node.get_output_name(),
                                                                model_node.intercept_data_name))
    return weld_block_node_list


def process_weld_block(weld_block_input_set, weld_block_aux_input_set, weld_block_output_set, weld_block_node_list,
                       future_nodes)\
        -> Tuple[str, List[str], List[str]]:
    """
    Helper function for graph_to_weld.  Creates a Willump statement for a block of Weld code given information about
    the code, its inputs, and its outputs.
    """
    for entry in weld_block_input_set:
        weld_block_node_list.insert(0, WillumpInputNode(entry))
    for entry in weld_block_aux_input_set:
        weld_block_node_list.insert(0, WillumpInputNode(entry))
    # Do not emit any output that are not needed in later blocks.
    for entry in weld_block_output_set.copy():
        appears_later = False
        for future_node in future_nodes:
            for node_input in future_node.get_in_nodes():
                input_name = node_input.get_output_name()
                if entry == input_name:
                    appears_later = True
        if not appears_later:
            weld_block_output_set.remove(entry)
    # Do optimization passes over the block.
    weld_block_node_list = pushing_model_pass(weld_block_node_list, weld_block_output_set)
    # Construct the Willump statements from the nodes.
    weld_block_node_list.append(WillumpMultiOutputNode(list(weld_block_output_set)))
    weld_str = ""
    for weld_node in weld_block_node_list:
        weld_str += weld_node.get_node_weld()
    return weld_str, list(weld_block_input_set), list(weld_block_output_set)


def graph_to_weld(graph: WillumpGraph) -> List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]]:
    """
    Convert a Willump graph into a sequence of Willump statements.

    A Willump statement consists of either a block of Python code or a block of Weld code with
    metadata.  The block of Python code is represented as an AST.  The block of Weld code
    is represented as a Weld string followed by a list of all its inputs followed by
    a list of all its outputs.
    """
    # We must first topologically sort the graph so that all of a node's inputs are computed before the node runs.
    sorted_nodes = topological_sort_graph(graph)
    weld_python_list: List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]] = []
    weld_block_input_set: Set[str] = set()
    weld_block_aux_input_set: Set[str] = set()
    weld_block_output_set: Set[str] = set()
    weld_block_node_list: List[WillumpGraphNode] = []
    for i, node in enumerate(sorted_nodes):
        if isinstance(node, WillumpPythonNode):
            if len(weld_block_node_list) > 0:
                processed_block_tuple = process_weld_block(weld_block_input_set, weld_block_aux_input_set,
                                                        weld_block_output_set, weld_block_node_list, sorted_nodes[i:])
                weld_python_list.append(processed_block_tuple)
                weld_block_input_set = set()
                weld_block_aux_input_set = set()
                weld_block_output_set = set()
                weld_block_node_list = []
            weld_python_list.append(node.get_python())
        elif isinstance(node, WillumpInputNode) or isinstance(node, WillumpOutputNode):
            pass
        else:
            for node_input in node.get_in_nodes():
                input_name = node_input.get_output_name()
                if "AUX_DATA" in input_name:
                    weld_block_aux_input_set.add(input_name)
                elif input_name not in weld_block_output_set:
                    weld_block_input_set.add(input_name)
            weld_block_output_set.add(node.get_output_name())
            weld_block_node_list.append(node)
    if len(weld_block_node_list) > 0:
        processed_block_tuple = process_weld_block(weld_block_input_set, weld_block_aux_input_set,
                                                   weld_block_output_set, weld_block_node_list, [sorted_nodes[-1]])
        weld_python_list.append(processed_block_tuple)
    return weld_python_list


def set_input_names(weld_program: str, args_list: List[str], aux_data: List[Tuple[int, WeldType]]) \
        -> str:
    for i, arg in enumerate(args_list):
        weld_program = weld_program.replace("WELD_INPUT_{0}__".format(arg), "_inp{0}".format(i))
    for i in range(len(aux_data)):
        weld_program = weld_program.replace("WELD_INPUT_AUX_DATA_{0}__".format(i),
                                            "_inp{0}".format(len(args_list) + i))
    return weld_program
