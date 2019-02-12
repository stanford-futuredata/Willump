from typing import List, Set, Tuple, MutableMapping, Optional
import typing
import ast

from willump.willump_utilities import *
import willump.evaluation.willump_graph_passes as wg_passes
import willump.evaluation.willump_cascades as w_cascades

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.willump_multioutput_node import WillumpMultiOutputNode


def process_weld_block(weld_block_input_set, weld_block_aux_input_set, weld_block_output_set, weld_block_node_list,
                       future_nodes, typing_map, batch, num_workers) \
        -> List[typing.Union[ast.AST, Tuple[List[str], List[str], List[List[str]]]]]:
    """
    Helper function for graph_to_weld.  Creates Willump statements for a block of Weld code given information about
    the code, its inputs, and its outputs.  Returns these Willump statements.
    """
    # Do not emit any output that are not needed in later blocks.
    for entry in weld_block_output_set.copy():
        appears_later = False
        for future_node in future_nodes:
            for input_name in future_node.get_in_names():
                if entry == input_name:
                    appears_later = True
        if not appears_later:
            weld_block_output_set.remove(entry)
    # Do optimization passes over the block.
    weld_block_node_list = wg_passes.pushing_model_pass(weld_block_node_list, weld_block_output_set, typing_map)
    csr_preprocess_nodes, csr_postprocess_nodes = \
        wg_passes.weld_csr_marshalling_pass(weld_block_input_set, weld_block_output_set, typing_map)
    pandas_preprocess_nodes, pandas_postprocess_nodes = wg_passes.weld_pandas_marshalling_pass(weld_block_input_set,
                                                                       weld_block_output_set, typing_map, batch)
    preprocess_nodes = csr_preprocess_nodes + pandas_preprocess_nodes
    postprocess_nodes = csr_postprocess_nodes + pandas_postprocess_nodes
    # Split Weld blocks into multiple threads.
    num_threads = num_workers + 1  # The main thread also does work.
    if num_threads > 1:
        threaded_statements_list = \
            wg_passes.multithreading_weld_blocks_pass(weld_block_node_list,
                                            weld_block_input_set, weld_block_output_set, num_threads)
    else:
        threaded_statements_list = [(weld_block_node_list, weld_block_input_set, weld_block_output_set)]
    # Append appropriate input and output nodes to each node list.
    for multithreaded_entry in threaded_statements_list:
        if len(multithreaded_entry) == 2:
            input_set, thread_list = multithreaded_entry
            for thread_entry in thread_list:
                weld_block_nodes, output_set = thread_entry
                for entry in input_set:
                    weld_block_nodes.insert(0, WillumpInputNode(entry))
                for entry in weld_block_aux_input_set:
                    weld_block_nodes.insert(0, WillumpInputNode(entry))
                weld_block_nodes.append(WillumpMultiOutputNode(list(output_set)))
        else:
            weld_block_nodes, input_set, output_set = multithreaded_entry
            for entry in input_set:
                weld_block_nodes.insert(0, WillumpInputNode(entry))
            for entry in weld_block_aux_input_set:
                weld_block_nodes.insert(0, WillumpInputNode(entry))
            weld_block_nodes.append(WillumpMultiOutputNode(list(output_set)))
    # Construct the statement list from the nodes.
    weld_string_nodes: List[Tuple[List[str], List[str], List[List[str]]]] = []
    for multithreaded_entry in threaded_statements_list:
        if len(multithreaded_entry) == 2:
            input_set, thread_list = multithreaded_entry
            weld_strings = []
            output_sets = []
            for thread_entry in thread_list:
                weld_block_nodes, output_set = thread_entry
                weld_str: str = ""
                for weld_node in weld_block_nodes:
                    weld_str += weld_node.get_node_weld()
                weld_strings.append(weld_str)
                output_sets.append(output_set)
            weld_string_nodes.append((weld_strings, list(input_set), output_sets))
        else:
            weld_block_nodes, input_set, output_set = multithreaded_entry
            weld_str: str = ""
            for weld_node in weld_block_nodes:
                weld_str += weld_node.get_node_weld()
            weld_string_nodes.append(([weld_str], list(input_set), [list(output_set)]))
    preprocess_python = list(map(lambda x: x.get_python(), preprocess_nodes))
    postprocess_python = list(map(lambda x: x.get_python(), postprocess_nodes))
    return preprocess_python + weld_string_nodes + postprocess_python


def graph_to_weld(graph: WillumpGraph, typing_map: MutableMapping[str, WeldType], training_cascades: Optional[dict],
    batch: bool = True, num_workers=0) -> \
        List[typing.Union[ast.AST, Tuple[List[str], List[str], List[List[str]]]]]:
    """
    Convert a Willump graph into a sequence of Willump statements.

    A Willump statement consists of either a block of Python code or a block of Weld code with
    metadata.  The block of Python code is represented as an AST.  The block of Weld code
    is represented as a Weld string followed by a list of all its inputs followed by
    a list of all its outputs.
    """
    # We must first topologically sort the graph so that all of a node's inputs are computed before the node runs.
    sorted_nodes = wg_passes.topological_sort_graph(graph)
    sorted_nodes = wg_passes.push_back_python_nodes_pass(sorted_nodes)
    sorted_nodes = wg_passes.async_python_functions_parallel_pass(sorted_nodes)
    wg_passes.model_input_identification_pass(sorted_nodes)
    if training_cascades is not None:
        sorted_nodes = w_cascades.model_cascade_pass(sorted_nodes, typing_map, training_cascades)
    weld_python_list: List[typing.Union[ast.AST, Tuple[List[str], List[str], List[str]]]] = []
    weld_block_input_set: Set[str] = set()
    weld_block_aux_input_set: Set[str] = set()
    weld_block_output_set: Set[str] = set()
    weld_block_node_list: List[WillumpGraphNode] = []
    for i, node in enumerate(sorted_nodes):
        if isinstance(node, WillumpPythonNode):
            if len(weld_block_node_list) > 0:
                processed_block_nodes = \
                    process_weld_block(weld_block_input_set, weld_block_aux_input_set, weld_block_output_set,
                                       weld_block_node_list, sorted_nodes[i:], typing_map, batch, num_workers)
                for processed_node in processed_block_nodes:
                    weld_python_list.append(processed_node)
                weld_block_input_set = set()
                weld_block_aux_input_set = set()
                weld_block_output_set = set()
                weld_block_node_list = []
            weld_python_list.append(node.get_python())
        elif isinstance(node, WillumpInputNode) or isinstance(node, WillumpOutputNode):
            pass
        else:
            for input_name in node.get_in_names():
                if "AUX_DATA" in input_name:
                    weld_block_aux_input_set.add(input_name)
                elif input_name not in weld_block_output_set:
                    weld_block_input_set.add(input_name)
            for output_name in node.get_output_names():
                weld_block_output_set.add(output_name)
            weld_block_node_list.append(node)
    if len(weld_block_node_list) > 0:
        processed_block_nodes = process_weld_block(weld_block_input_set, weld_block_aux_input_set,
                                                   weld_block_output_set, weld_block_node_list,
                                                   [sorted_nodes[-1]], typing_map, batch, num_workers)
        for processed_node in processed_block_nodes:
            weld_python_list.append(processed_node)
    return weld_python_list


def set_input_names(weld_program: str, args_list: List[str], aux_data: List[Tuple[int, WeldType]]) \
        -> str:
    for i, arg in enumerate(args_list):
        weld_program = weld_program.replace("WELD_INPUT_{0}__".format(arg), "_inp{0}".format(i))
    for i in range(len(aux_data)):
        weld_program = weld_program.replace("WELD_INPUT_AUX_DATA_{0}__".format(i),
                                            "_inp{0}".format(len(args_list) + i))
    return weld_program
