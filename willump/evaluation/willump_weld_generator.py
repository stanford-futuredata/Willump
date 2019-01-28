from typing import MutableMapping, List, Set, Tuple, Mapping, Optional
import typing
import ast
import copy

from weld.types import *
from willump import panic

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.willump_multioutput_node import WillumpMultiOutputNode
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.combine_linear_regression_node import CombineLinearRegressionNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.willump_hash_join_node import WillumpHashJoinNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode


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


def push_back_python_nodes_pass(sorted_nodes: List[WillumpGraphNode]) -> List[WillumpGraphNode]:
    """
    Greedily push Python nodes and input nodes back towards the start of the sorted_nodes list while maintaining
    topological sorting.  This maximizes the length of Weld blocks.
    """
    for i in range(len(sorted_nodes)):
        if isinstance(sorted_nodes[i], WillumpPythonNode) or isinstance(sorted_nodes[i], WillumpInputNode):
            python_node = sorted_nodes[i]
            input_names: List[str] = list(map(lambda x: x.get_output_name(), python_node.get_in_nodes()))
            good_index: int = i
            for j in range(i - 1, 0 - 1, -1):
                if sorted_nodes[j].get_output_name() in input_names:
                    break
                else:
                    good_index = j
            sorted_nodes.remove(python_node)
            sorted_nodes.insert(good_index, python_node)
    return sorted_nodes


def pushing_model_pass(weld_block_node_list, weld_block_output_set, typing_map) -> List[WillumpGraphNode]:
    """
    Optimization pass that pushes a linear model into its input nodes in a block when possible.  Assumes block contains
    at most one model node, which must be the last node in the block.

    # TODO:  Support more classes of models.
    """

    def node_is_transformable(node):
        """
        This node can be freely transformed.
        """
        return node in weld_block_node_list and node.get_output_name() not in weld_block_output_set

    model_node = weld_block_node_list[-1]
    if not isinstance(model_node, LinearRegressionNode):
        return weld_block_node_list
    # A stack containing the next nodes we want to push the model through and what their indices are in the model.
    current_node_stack: List[Tuple[WillumpGraphNode, Tuple[int, int], Optional[Mapping[str, int]]]] = \
        [(model_node.get_in_nodes()[0], (0, model_node.input_width), None)]
    # Nodes through which the model has been pushed which are providing output.
    nodes_to_sum: List[WillumpGraphNode] = []
    while len(current_node_stack) > 0:
        input_node, (index_start, index_end), curr_selection_map = current_node_stack.pop()
        if node_is_transformable(input_node):
            if isinstance(input_node, ArrayCountVectorizerNode) or isinstance(input_node, ArrayTfIdfNode):
                input_node.push_model("linear", (model_node.weights_data_name,), index_start)
                nodes_to_sum.append(input_node)
            elif isinstance(input_node, StackSparseNode):
                if all(node_is_transformable(stacked_node) for stacked_node in input_node.get_in_nodes()):
                    stack_start_index = index_start
                    for stacked_node in input_node.get_in_nodes():
                        output_width = stacked_node.output_width
                        current_node_stack.append(
                            (stacked_node, (stack_start_index, stack_start_index + output_width), curr_selection_map))
                        stack_start_index += output_width
                    weld_block_node_list.remove(input_node)
                else:
                    # TODO:  Push directly onto the stacking node.
                    panic("Pushing onto stacking node not implemented")
            elif isinstance(input_node, PandasColumnSelectionNode):
                selected_columns: List[str] = input_node.selected_columns
                assert (len(selected_columns) == index_end - index_start)
                selection_map: Mapping[str, int] = {col: index_start + i for i, col in enumerate(selected_columns)}
                selection_input = input_node.get_in_nodes()[0]
                if node_is_transformable(selection_input):
                    current_node_stack.append((selection_input, (index_start, index_end), selection_map))
                    weld_block_node_list.remove(input_node)
                else:
                    # TODO:  Push directly onto the selection node.
                    panic("Pushing onto selection node not implemented")
            elif isinstance(input_node, WillumpHashJoinNode):
                join_left_columns = input_node.left_df_type.column_names
                join_right_columns = input_node.right_df_type.column_names
                output_columns = join_left_columns + join_right_columns
                if curr_selection_map is None:
                    curr_selection_map = {col: index_start + i for i, col in enumerate(output_columns)}
                join_left_input = input_node.get_in_nodes()[0]
                if node_is_transformable(join_left_input) and not (isinstance(join_left_input,
                                                                              WillumpHashJoinNode) and input_node.join_col_name in
                                                                   join_left_input.right_df_type.column_names):
                    pushed_map = {}
                    next_map = {}
                    for col in curr_selection_map.keys():
                        if col in join_right_columns:
                            pushed_map[col] = curr_selection_map[col]
                        else:
                            next_map[col] = curr_selection_map[col]
                    current_node_stack.append((join_left_input, (index_start, index_end), next_map))
                    # TODO:  Make this input name discovery recurse for arbitrarily deep joins.
                    new_input_name = join_left_input.input_array_string_name
                else:
                    pushed_map = {}
                    for col in curr_selection_map.keys():
                        if col in output_columns:
                            pushed_map[col] = curr_selection_map[col]
                    new_input_name = input_node.input_array_string_name
                input_node.push_model("linear", (model_node.weights_data_name,), pushed_map,
                                      new_input_name)
                nodes_to_sum.append(input_node)
    if len(nodes_to_sum) > 0:
        weld_block_node_list.remove(model_node)
        weld_block_node_list.append(CombineLinearRegressionNode(input_nodes=nodes_to_sum,
                                                                output_name=model_node.get_output_name(),
                                                                intercept_data_name=model_node.intercept_data_name,
                                                                output_type=typing_map[model_node.get_output_name()]))
    return weld_block_node_list


def weld_pandas_marshalling_pass(weld_block_input_set: Set[str], weld_block_output_set: Set[str],
                                 typing_map: Mapping[str, WeldType], batch: bool) \
        -> Tuple[List[WillumpPythonNode], List[WillumpPythonNode]]:
    """
    Processing pass creating Python code to marshall Pandas into a representation (struct of vec columns) Weld can
    understand, then convert that struct back to Pandas.

    # TODO:  Reconstruct the original index properly.
    """
    pandas_input_processing_nodes: List[WillumpPythonNode] = []
    pandas_post_processing_nodes: List[WillumpPythonNode] = []
    for input_name in weld_block_input_set:
        input_type = typing_map[input_name]
        if isinstance(input_type, WeldPandas):
            df_temp_name = "%s__df_temp" % input_name
            if batch:
                pandas_glue_python_args = ""
                for column in input_type.column_names:
                    pandas_glue_python_args += "%s['%s'].values," % (input_name, column)
                pandas_glue_python = "%s = (%s)" % (input_name, pandas_glue_python_args)
            else:
                pandas_glue_python = "%s = tuple(%s.values[0])" % (input_name, input_name)
            pandas_glue_ast: ast.Module = \
                ast.parse(pandas_glue_python, "exec")
            pandas_input_node = WillumpPythonNode(pandas_glue_ast.body[0], input_name, [])
            pandas_input_processing_nodes.append(pandas_input_node)
    for output_name in weld_block_output_set:
        output_type = typing_map[output_name]
        if isinstance(output_type, WeldPandas):
            df_creation_arguments = ""
            for i, column_name in enumerate(output_type.column_names):
                if batch:
                    df_creation_arguments += "'%s' : %s[%d]," % (column_name, output_name, i)
                else:
                    df_creation_arguments += "'%s' : [%s[%d]]," % (column_name, output_name, i)
            df_creation_statement = "%s = pd.DataFrame({%s})" % (output_name, df_creation_arguments)
            df_creation_ast: ast.Module = \
                ast.parse(df_creation_statement, "exec")
            df_creation_node = WillumpPythonNode(df_creation_ast.body[0], output_name, [])
            pandas_post_processing_nodes.append(df_creation_node)
    return pandas_input_processing_nodes, pandas_post_processing_nodes


def process_weld_block(weld_block_input_set, weld_block_aux_input_set, weld_block_output_set, weld_block_node_list,
                       future_nodes, typing_map, batch) \
        -> List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]]:
    """
    Helper function for graph_to_weld.  Creates Willump statements for a block of Weld code given information about
    the code, its inputs, and its outputs.  Returns these Willump statements.
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
    weld_block_node_list = pushing_model_pass(weld_block_node_list, weld_block_output_set, typing_map)
    preprocess_nodes, postprocess_nodes = weld_pandas_marshalling_pass(weld_block_input_set,
                                                                       weld_block_output_set, typing_map, batch)
    # Construct the Willump statements from the nodes.
    weld_block_node_list.append(WillumpMultiOutputNode(list(weld_block_output_set)))
    weld_str = ""
    for weld_node in weld_block_node_list:
        weld_str += weld_node.get_node_weld()
    preprocess_python = list(map(lambda x: x.get_python(), preprocess_nodes))
    postprocess_python = list(map(lambda x: x.get_python(), postprocess_nodes))
    return preprocess_python + [(weld_str, list(weld_block_input_set), list(weld_block_output_set))] \
           + postprocess_python


def graph_to_weld(graph: WillumpGraph, typing_map: Mapping[str, WeldType], batch: bool = True) -> \
        List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]]:
    """
    Convert a Willump graph into a sequence of Willump statements.

    A Willump statement consists of either a block of Python code or a block of Weld code with
    metadata.  The block of Python code is represented as an AST.  The block of Weld code
    is represented as a Weld string followed by a list of all its inputs followed by
    a list of all its outputs.
    """
    # We must first topologically sort the graph so that all of a node's inputs are computed before the node runs.
    sorted_nodes = topological_sort_graph(graph)
    sorted_nodes = push_back_python_nodes_pass(sorted_nodes)
    weld_python_list: List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]] = []
    weld_block_input_set: Set[str] = set()
    weld_block_aux_input_set: Set[str] = set()
    weld_block_output_set: Set[str] = set()
    weld_block_node_list: List[WillumpGraphNode] = []
    for i, node in enumerate(sorted_nodes):
        if isinstance(node, WillumpPythonNode):
            if len(weld_block_node_list) > 0:
                processed_block_nodes = \
                    process_weld_block(weld_block_input_set, weld_block_aux_input_set,
                                       weld_block_output_set, weld_block_node_list, sorted_nodes[i:], typing_map, batch)
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
            for node_input in node.get_in_nodes():
                input_name = node_input.get_output_name()
                if "AUX_DATA" in input_name:
                    weld_block_aux_input_set.add(input_name)
                elif input_name not in weld_block_output_set:
                    weld_block_input_set.add(input_name)
            weld_block_output_set.add(node.get_output_name())
            weld_block_node_list.append(node)
    if len(weld_block_node_list) > 0:
        processed_block_nodes = process_weld_block(weld_block_input_set, weld_block_aux_input_set,
                                                   weld_block_output_set, weld_block_node_list,
                                                   [sorted_nodes[-1]], typing_map, batch)
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
