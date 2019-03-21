import ast
import copy
import math
import typing
from collections import OrderedDict
from typing import MutableMapping, List, Set, Tuple, Mapping, Optional

from willump import *
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from willump.graph.combine_linear_regression_node import CombineLinearRegressionNode
from willump.graph.hash_join_node import WillumpHashJoinNode
from willump.graph.identity_node import IdentityNode
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.pandas_series_concatenation_node import PandasSeriesConcatenationNode
from willump.graph.pandas_to_dense_matrix_node import PandasToDenseMatrixNode
from willump.graph.reshape_node import ReshapeNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.train_test_split_node import TrainTestSplitNode
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_model_node import WillumpModelNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import *


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
            input_names: List[str] = python_node.get_in_names()
            good_index: int = i
            for j in range(i - 1, 0 - 1, -1):
                if any(output_name in input_names for output_name in sorted_nodes[j].get_output_names()):
                    break
                else:
                    good_index = j
            sorted_nodes.remove(python_node)
            sorted_nodes.insert(good_index, python_node)
    return sorted_nodes


def find_dataframe_base_node(df_node: WillumpGraphNode,
                             nodes_to_base_map: MutableMapping[WillumpGraphNode, WillumpHashJoinNode]) \
        -> WillumpHashJoinNode:
    """
    If a model's input contains a sequence of independent joins of metadata onto the same table of data,
    identify the first join onto that data table.
    """
    if df_node in nodes_to_base_map:
        return nodes_to_base_map[df_node]
    base_discovery_node = df_node
    join_cols_set = set()
    touched_nodes = []
    while True:
        assert (isinstance(base_discovery_node, WillumpHashJoinNode))
        for entry in base_discovery_node.join_col_names:
            join_cols_set.add(entry)
        touched_nodes.append(base_discovery_node)
        base_left_input = base_discovery_node.get_in_nodes()[0]
        if isinstance(base_left_input, WillumpHashJoinNode) and not any(
                join_col_name in base_left_input.right_df_row_type.column_names for join_col_name in join_cols_set):
            base_discovery_node = base_left_input
        else:
            break
    for df_node in touched_nodes:
        nodes_to_base_map[df_node] = base_discovery_node
    assert (isinstance(base_discovery_node, WillumpHashJoinNode))
    return base_discovery_node


def model_input_identification_pass(sorted_nodes: List[WillumpGraphNode]) -> None:
    """
    Identify the model, if there is one, in a program.  Set its model inputs.  Do not modify the nodes in
    any other way.

    Assumption:  There is one model in the program.
    """
    nodes_to_base_map: MutableMapping[WillumpGraphNode, WillumpGraphNode] = {}
    for node in sorted_nodes:
        if isinstance(node, WillumpModelNode):
            model_node = node
            break
    else:
        return
    # A stack containing the next nodes we want to examine and what their indices are in the model.
    current_node_stack: List[Tuple[WillumpGraphNode, Tuple[int, int], Optional[Mapping[str, int]]]] = \
        [(model_node.get_in_nodes()[0], (0, model_node.input_width), None)]
    model_inputs: MutableMapping[WillumpGraphNode, typing.Union[Tuple[int, int], Mapping[str, int]]] = {}
    while len(current_node_stack) > 0:
        input_node, (index_start, index_end), curr_selection_map = current_node_stack.pop()
        if isinstance(input_node, ArrayCountVectorizerNode) or isinstance(input_node, ArrayTfIdfNode):
            model_inputs[input_node] = (index_start, index_end)
        elif isinstance(input_node, StackSparseNode):
            stack_start_index = index_start
            for stacked_node in input_node.get_in_nodes():
                try:
                    output_width = stacked_node.output_width
                except AttributeError:
                    assert (len(stacked_node.get_output_types()) == 1)
                    node_output_type = stacked_node.get_output_types()[0]
                    assert (isinstance(node_output_type, WeldCSR))
                    output_width = node_output_type.width
                current_node_stack.append(
                    (stacked_node, (stack_start_index, stack_start_index + output_width), curr_selection_map))
                stack_start_index += output_width
        elif isinstance(input_node, PandasColumnSelectionNode):
            selected_columns: List[str] = input_node.selected_columns
            assert (len(selected_columns) == index_end - index_start)
            selection_map: Mapping[str, int] = {col: index_start + i for i, col in enumerate(selected_columns)}
            selection_input = input_node.get_in_nodes()[0]
            current_node_stack.append((selection_input, (index_start, index_end), selection_map))
        elif isinstance(input_node, PandasSeriesConcatenationNode):
            for node_input_node, input_type in zip(input_node.get_in_nodes(), input_node.input_types):
                node_map = {}
                for col in curr_selection_map.keys():
                    if col in input_type.column_names:
                        node_map[col] = curr_selection_map[col]
                current_node_stack.append((node_input_node, (index_start, index_end), node_map))
        elif isinstance(input_node, WillumpHashJoinNode):
            join_left_columns = input_node.left_df_type.column_names
            join_right_columns = input_node.right_df_row_type.column_names
            output_columns = join_left_columns + join_right_columns
            if curr_selection_map is None:
                curr_selection_map = {col: index_start + i for i, col in enumerate(output_columns)}
            join_base_node = find_dataframe_base_node(input_node, nodes_to_base_map)
            pushed_map = {}
            next_map = {}
            for col in curr_selection_map.keys():
                if col in join_right_columns:
                    pushed_map[col] = curr_selection_map[col]
                else:
                    next_map[col] = curr_selection_map[col]
            join_left_input = input_node.get_in_nodes()[0]
            model_inputs[input_node] = pushed_map
            if join_base_node is not input_node:
                current_node_stack.append((join_left_input, (index_start, index_end), next_map))
            else:
                model_inputs[join_left_input] = next_map
        elif isinstance(input_node, IdentityNode) or isinstance(input_node, ReshapeNode) \
                or isinstance(input_node, PandasToDenseMatrixNode) or isinstance(input_node, TrainTestSplitNode):
            node_input_node = input_node.get_in_nodes()[0]
            current_node_stack.append((node_input_node, (index_start, index_end), curr_selection_map))
        elif isinstance(input_node, WillumpInputNode) or isinstance(input_node, WillumpPythonNode):
            if curr_selection_map is not None:
                model_inputs[input_node] = curr_selection_map
            else:
                model_inputs[input_node] = (index_start, index_end)
        else:
            panic("Unrecognized node found when processing model inputs %s" % input_node.__repr__())
    model_node.set_model_inputs(model_inputs)
    return


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
        return node in weld_block_node_list and not \
            any(output_name in weld_block_output_set for output_name in node.get_output_names())

    nodes_to_base_map = {}
    model_node = weld_block_node_list[-1]
    if (not isinstance(model_node, LinearRegressionNode)) or model_node.predict_proba or \
            any(isinstance(w_node, WillumpHashJoinNode) for w_node in weld_block_node_list):
        return weld_block_node_list
    model_inputs: Mapping[WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]] = model_node.get_model_inputs()
    # A stack containing the next nodes we want to push the model through.
    current_node_stack: List[WillumpGraphNode] = model_node.get_in_nodes()
    # Nodes through which the model has been pushed which are providing output.
    nodes_to_sum: List[WillumpGraphNode] = []
    while len(current_node_stack) > 0:
        input_node = current_node_stack.pop()
        if node_is_transformable(input_node):
            if isinstance(input_node, ArrayCountVectorizerNode) or isinstance(input_node, ArrayTfIdfNode):
                index_start, _ = model_inputs[input_node]
                input_node.push_model("linear", (model_node.weights_data_name,), index_start)
                # Set that node's output's new type.
                typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
                nodes_to_sum.append(input_node)
            elif isinstance(input_node, StackSparseNode):
                if all(node_is_transformable(stacked_node) for stacked_node in input_node.get_in_nodes()):
                    for stacked_node in input_node.get_in_nodes():
                        current_node_stack.append(stacked_node)
                    weld_block_node_list.remove(input_node)
                else:
                    index_start = 0  # TODO:  Update
                    input_node.push_model("linear", (model_node.weights_data_name,), index_start)
                    # Set that node's output's new type.
                    typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
                    nodes_to_sum.append(input_node)
            # elif isinstance(input_node, PandasColumnSelectionNode):
            #     selection_input = input_node.get_in_nodes()[0]
            #     if node_is_transformable(selection_input):
            #         current_node_stack.append(selection_input)
            #         weld_block_node_list.remove(input_node)
            #     else:
            #         selection_map: Mapping[str, int] = model_inputs[input_node]
            #         assert (isinstance(selection_map, Mapping))
            #         input_node.push_model("linear", (model_node.weights_data_name,), selection_map)
            #         typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
            #         nodes_to_sum.append(input_node)
            # elif isinstance(input_node, WillumpHashJoinNode):
            #     join_left_input_node = input_node.get_in_nodes()[0]
            #     if join_left_input_node in model_inputs:
            #         current_node_stack.append(join_left_input_node)
            #     pushed_map: Mapping[str, int] = model_inputs[input_node]
            #     if len(pushed_map) > 0:
            #         assert (isinstance(pushed_map, Mapping))
            #         base_node = find_dataframe_base_node(input_node, nodes_to_base_map)
            #         input_node.left_input_name = base_node.left_input_name
            #         input_node.push_model("linear", (model_node.weights_data_name,), pushed_map)
            #         # Set that node's output's new type.
            #         typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
            #         nodes_to_sum.append(input_node)
            #     else:
            #         weld_block_node_list.remove(input_node)
            elif isinstance(input_node, IdentityNode):
                current_node_stack.append(input_node.get_in_nodes()[0])
                weld_block_node_list.remove(input_node)
    if len(nodes_to_sum) > 0:
        weld_block_node_list.remove(model_node)
        weld_block_node_list.append(CombineLinearRegressionNode(input_nodes=nodes_to_sum,
                                                                output_name=model_node.get_output_name(),
                                                                intercept_data_name=model_node.intercept_data_name,
                                                                output_type=typing_map[model_node.get_output_name()],
                                                                regression=model_node._regression))
    return weld_block_node_list


def weld_pandas_marshalling_pass(weld_block_input_set: Set[str], weld_block_output_set: Set[str],
                                 typing_map: Mapping[str, WeldType], batch) \
        -> Tuple[List[WillumpPythonNode], List[WillumpPythonNode]]:
    """
    Processing pass creating Python code to marshall Pandas into a representation (struct of vec columns) Weld can
    understand, then convert that struct back to Pandas.

    # TODO:  Reconstruct the original index properly.
    """
    pandas_input_processing_nodes: List[WillumpPythonNode] = []
    pandas_post_processing_nodes: List[WillumpPythonNode] = []
    if not batch:  # TODO:  This is a hack, fix it
        return pandas_input_processing_nodes, pandas_post_processing_nodes
    for input_name in weld_block_input_set:
        input_type = typing_map[input_name]
        if isinstance(input_type, WeldPandas):
            df_temp_name = "%s__df_temp" % input_name
            # Strip line numbers from variable name.
            stripped_input_name = strip_linenos_from_var(input_name)
            pandas_glue_python_args = ""
            for column, field_type in zip(input_type.column_names, input_type.field_types):
                if isinstance(field_type, WeldVec) and isinstance(field_type.elemType, WeldStr):
                    pandas_glue_python_args += "list(%s['%s'].values)," % (stripped_input_name, column)
                else:
                    pandas_glue_python_args += "%s['%s'].values," % (stripped_input_name, column)
            pandas_glue_python = "%s, %s = (%s), %s" % (
                stripped_input_name, df_temp_name, pandas_glue_python_args, stripped_input_name)
            pandas_glue_ast: ast.Module = \
                ast.parse(pandas_glue_python, "exec")
            pandas_input_node = WillumpPythonNode(python_ast=pandas_glue_ast.body[0], input_names=[],
                                                  output_names=[input_name], output_types=[], in_nodes=[])
            pandas_input_processing_nodes.append(pandas_input_node)
            reversion_python = "%s = %s" % (stripped_input_name, df_temp_name)
            reversion_python_ast = ast.parse(reversion_python, "exec")
            reversion_python_node = WillumpPythonNode(python_ast=reversion_python_ast.body[0],
                                                      input_names=[df_temp_name],
                                                      output_names=[stripped_input_name], output_types=[], in_nodes=[])
            if stripped_input_name not in map(strip_linenos_from_var, weld_block_output_set):
                pandas_post_processing_nodes.append(reversion_python_node)

    for output_name in weld_block_output_set:
        output_type = typing_map[output_name]
        stripped_output_name = strip_linenos_from_var(output_name)
        if isinstance(output_type, WeldPandas):
            df_creation_arguments = ""
            for i, column_name in enumerate(output_type.column_names):
                df_creation_arguments += "'%s' : %s[%d]," % (column_name, stripped_output_name, i)
            df_creation_statement = "%s = pd.DataFrame({%s})" % (stripped_output_name, df_creation_arguments)
            df_creation_ast: ast.Module = \
                ast.parse(df_creation_statement, "exec")
            df_creation_node = WillumpPythonNode(python_ast=df_creation_ast.body[0], input_names=[],
                                                 output_names=[output_name], output_types=[], in_nodes=[])
            pandas_post_processing_nodes.append(df_creation_node)
    return pandas_input_processing_nodes, pandas_post_processing_nodes


def weld_pandas_series_marshalling_pass(weld_block_input_set: Set[str], weld_block_output_set: Set[str],
                                        typing_map: Mapping[str, WeldType]) \
        -> Tuple[List[WillumpPythonNode], List[WillumpPythonNode]]:
    """
    Processing pass creating Python code to marshall Pandas Series into a representation (numpy array) Weld can
    understand.

    # TODO:  Also convert back to Pandas series.
    """
    pandas_input_processing_nodes: List[WillumpPythonNode] = []
    pandas_post_processing_nodes: List[WillumpPythonNode] = []
    for input_name in weld_block_input_set:
        input_type = typing_map[input_name]
        if isinstance(input_type, WeldSeriesPandas):
            # Strip line numbers from variable name.
            stripped_input_name = strip_linenos_from_var(input_name)
            series_glue_python = "%s = %s.values" % (stripped_input_name, stripped_input_name)
            series_glue_ast: ast.Module = \
                ast.parse(series_glue_python, "exec")
            series_input_node = WillumpPythonNode(python_ast=series_glue_ast.body[0], input_names=[],
                                                  output_names=[input_name], output_types=[], in_nodes=[])
            pandas_input_processing_nodes.append(series_input_node)
    return pandas_input_processing_nodes, pandas_post_processing_nodes


def weld_csr_marshalling_pass(weld_block_input_set: Set[str], weld_block_output_set: Set[str],
                              typing_map: Mapping[str, WeldType]) \
        -> Tuple[List[WillumpPythonNode], List[WillumpPythonNode]]:
    """
    Processing pass creating Python code to marshall CSR-format sparse matrices from Weld into Python.
    """
    csr_input_processing_nodes: List[WillumpPythonNode] = []
    csr_post_processing_nodes: List[WillumpPythonNode] = []
    for input_name in weld_block_input_set:
        input_type = typing_map[input_name]
        stripped_input_name = strip_linenos_from_var(input_name)
        store_weld_form_name = "weld_csr_" + stripped_input_name
        temp_name = "temp_" + stripped_input_name
        if isinstance(input_type, WeldCSR):
            csr_marshall = """{0}, {2} = csr_marshall({0}), {0}\n""".format(
                stripped_input_name, store_weld_form_name, temp_name)
            csr_creation_ast: ast.Module = \
                ast.parse(csr_marshall, "exec")
            csr_creation_node = WillumpPythonNode(python_ast=csr_creation_ast.body[0], input_names=[],
                                                  output_names=[], output_types=[], in_nodes=[])
            csr_input_processing_nodes.append(csr_creation_node)
            car_unmarshaller = """{0} = {1}\n""".format(stripped_input_name, temp_name)
            csr_unmarshaller_ast: ast.Module = \
                ast.parse(car_unmarshaller, "exec")
            csr_unmarshaller_node = WillumpPythonNode(python_ast=csr_unmarshaller_ast.body[0], input_names=[],
                                                      output_names=[], output_types=[], in_nodes=[])
            csr_post_processing_nodes.append(csr_unmarshaller_node)
    for output_name in weld_block_output_set:
        output_type = typing_map[output_name]
        stripped_output_name = strip_linenos_from_var(output_name)
        store_weld_form_name = "weld_csr_" + stripped_output_name
        if isinstance(output_type, WeldCSR):
            csr_marshall = """{0}, {1} = scipy.sparse.csr_matrix(({0}[2], ({0}[0], {0}[1])), shape=({0}[3], {0}[4])), {0}\n""".format(
                stripped_output_name, store_weld_form_name)
            csr_creation_ast: ast.Module = \
                ast.parse(csr_marshall, "exec")
            csr_creation_node = WillumpPythonNode(python_ast=csr_creation_ast.body[0], input_names=[],
                                                  output_names=[output_name], output_types=[], in_nodes=[])
            csr_post_processing_nodes.append(csr_creation_node)
    return csr_input_processing_nodes, csr_post_processing_nodes


def multithreading_weld_blocks_pass(weld_block_node_list: List[WillumpGraphNode],
                                    weld_block_input_set: Set[str],
                                    weld_block_output_set: Set[str],
                                    num_threads: int) \
        -> List[typing.Union[Tuple[List[WillumpGraphNode], Set[str], Set[str]],
                             Tuple[Set[str], List[Tuple[List[WillumpGraphNode], Set[str]]]]]]:
    """
    Identify opportunities for multithreading in the node list and parallelize them.  Return a list of sequential
    Weld blocks (Tuples of node lists, inputs, and outputs) or parallel node blocks (inputs, then a list of tuples
    of node lists and outputs).

    TODO:  Only parallelizes node blocks ending in a CombineLinearRegressionNode.

    TODO:  Assumes subgraphs rooted in inputs to a CombineLinearRegressionNode and in weld_block_node_list are disjoint.
    """
    combine_node = weld_block_node_list[-1]
    if not isinstance(combine_node, CombineLinearRegressionNode):
        return [(weld_block_node_list, weld_block_input_set, weld_block_output_set)]
    input_nodes = combine_node.get_in_nodes()
    thread_list: List[Tuple[List[WillumpGraphNode], Set[str]]] = []
    for model_input_node in input_nodes:
        # The graph below model_input_node, topologically sorted.
        sorted_nodes: List[WillumpGraphNode] = topological_sort_graph(WillumpGraph(model_input_node))
        # Only the nodes from that graph which are in our block.
        sorted_nodes = list(filter(lambda x: x in weld_block_node_list, sorted_nodes))
        output_set: Set[str] = set()
        for node in sorted_nodes:
            output_name = node.get_output_name()
            if output_name in weld_block_output_set or node is model_input_node:
                output_set.add(output_name)
        thread_list.append((sorted_nodes, output_set))
    # Coalesce if more jobs than threads.
    if len(thread_list) > num_threads:
        thread_list_length = len(thread_list)  # To avoid Python weirdness.
        thread_list_index = 0
        while thread_list_length > num_threads:
            assert (thread_list_index < thread_list_length)
            graph_nodes_one, outputs_one = thread_list[thread_list_index]
            graph_nodes_two, outputs_two = thread_list[(thread_list_index + 1) % thread_list_length]
            graph_nodes_comb = graph_nodes_one + graph_nodes_two
            outputs_comb = outputs_one.union(outputs_two)
            new_entry: List[Tuple[List[WillumpGraphNode], Set[str]]] = [(graph_nodes_comb, outputs_comb)]
            if (thread_list_index + 1) % thread_list_length != 0:
                thread_list = thread_list[0:thread_list_index] + new_entry + thread_list[thread_list_index + 2:]
            else:
                thread_list = new_entry + thread_list[1:-1]
            thread_list_length -= 1
            thread_list_index = (thread_list_index + 1) % thread_list_length
    parallel_entry: Tuple[Set[str], List[Tuple[List[WillumpGraphNode], Set[str]]]] = (weld_block_input_set, thread_list)
    combiner_input_names = set(combine_node.get_in_names())
    combiner_output_name = {combine_node.get_output_name()}
    sequential_entry: Tuple[List[WillumpGraphNode], Set[str], Set[str]] = ([combine_node],
                                                                           combiner_input_names, combiner_output_name)
    return [parallel_entry, sequential_entry]


def async_python_functions_parallel_pass(sorted_nodes: List[WillumpGraphNode]) \
        -> List[WillumpGraphNode]:
    """
    Run all asynchronous Python nodes in separate threads.
    """

    def find_pyblock_after_n(n: int) -> Optional[Tuple[int, int]]:
        start_index = None
        end_index = None
        for i, statement in enumerate(sorted_nodes):
            if i > n and isinstance(statement, WillumpPythonNode):
                start_index = i
                break
        if start_index is None:
            return None
        for i, statement in enumerate(sorted_nodes):
            if i > start_index and not isinstance(statement, WillumpPythonNode):
                end_index = i
                break
        if end_index is None:
            end_index = len(sorted_nodes)
        return start_index, end_index

    pyblock_start_end = find_pyblock_after_n(-1)
    while pyblock_start_end is not None:
        pyblock_start, pyblock_end = pyblock_start_end
        pyblock: List[WillumpPythonNode] = sorted_nodes[pyblock_start:pyblock_end]
        async_nodes = []
        for node in pyblock:
            if node.is_async_node:
                async_nodes.append(node)
        for async_node in async_nodes:
            assert (len(async_node.get_output_names()) == 1)
            async_node_output_name: str = async_node.get_output_names()[0]
            async_node_index = pyblock.index(async_node)
            input_names: List[str] = async_node.get_in_names()
            first_legal_index: int = async_node_index
            for j in range(async_node_index - 1, 0 - 1, -1):
                if any(output_name in input_names for output_name in pyblock[j].get_output_names()):
                    break
                else:
                    first_legal_index = j
            pyblock.remove(async_node)
            async_node_ast: ast.Assign = async_node.get_python()
            assert (isinstance(async_node_ast.value, ast.Call))
            assert (isinstance(async_node_ast.value.func, ast.Name))
            # output = executor.submit(original func names, original func args...)
            new_call = ast.Call()
            new_call.keywords = []
            new_call.func = ast.Attribute()
            new_call.func.attr = "submit"
            new_call.func.ctx = ast.Load()
            new_call.func.value = ast.Name()
            new_call.func.value.id = WILLUMP_THREAD_POOL_EXECUTOR
            new_call.func.value.ctx = ast.Load()
            new_args = async_node_ast.value.args
            new_first_arg = ast.Name()
            new_first_arg.id = async_node_ast.value.func.id
            new_first_arg.ctx = ast.Load()
            new_args.insert(0, new_first_arg)
            new_call.args = new_args
            async_node_ast.value = new_call
            executor_async_node = WillumpPythonNode(python_ast=async_node_ast, input_names=async_node.get_in_names(),
                                                    output_names=async_node.get_output_names(),
                                                    output_types=async_node.get_output_types(),
                                                    in_nodes=async_node.get_in_nodes())
            pyblock.insert(first_legal_index, executor_async_node)
            last_legal_index = first_legal_index + 1
            for j in range(first_legal_index + 1, len(pyblock)):
                curr_node = pyblock[j]
                node_inputs = curr_node.get_in_names()
                if async_node_output_name in node_inputs:
                    break
                else:
                    last_legal_index = j
            result_python = "%s = %s.result()" % (strip_linenos_from_var(async_node_output_name),
                                                  strip_linenos_from_var(async_node_output_name))
            result_ast: ast.Module = \
                ast.parse(result_python, "exec")
            pandas_input_node = WillumpPythonNode(python_ast=result_ast.body[0],
                                                  input_names=[async_node_output_name],
                                                  output_names=async_node.get_output_names(),
                                                  output_types=async_node.get_output_types(),
                                                  in_nodes=[executor_async_node])
            pyblock.insert(last_legal_index, pandas_input_node)
        sorted_nodes = sorted_nodes[:pyblock_start] + pyblock + sorted_nodes[pyblock_end:]
        pyblock_start_end = find_pyblock_after_n(pyblock_end)

    return sorted_nodes


def cache_python_block_pass(sorted_nodes: List[WillumpGraphNode], willump_cache_dict: dict,
                            max_cache_size: Optional[int]) -> List[WillumpGraphNode]:
    """
    Detect cacheable functions and cache them by replacing calls to them with calls to the caching function.
    """
    node_num = 0
    for i, entry in enumerate(sorted_nodes):
        if isinstance(entry, WillumpPythonNode) and entry.is_cached_node:
            entry_ast: ast.Assign = entry.get_python()
            assert (isinstance(entry_ast, ast.Assign))
            value = entry_ast.value
            assert (isinstance(value, ast.Call))
            assert (isinstance(value.func, ast.Name))
            assert (len(entry_ast.targets) == 1)
            if all(isinstance(arg_entry, ast.Name) for arg_entry in value.args) \
                    and isinstance(entry_ast.targets[0], ast.Name):
                target_name = entry_ast.targets[0].id
                func_name = value.func.id
                arg_string = ""
                for arg in value.args:
                    arg_string += "%s," % arg.id
                cache_python = \
                    """%s = willump_cache(%s, (%s), %s, %d)""" % \
                    (target_name, func_name, arg_string, WILLUMP_CACHE_NAME, node_num)
                cache_ast: ast.Module = ast.parse(cache_python, "exec")
                cache_node = WillumpPythonNode(python_ast=cache_ast.body[0], input_names=entry.get_in_names(),
                                               output_names=entry.get_output_names(),
                                               output_types=entry.get_output_types(),
                                               in_nodes=entry.get_in_nodes())
                sorted_nodes[i] = cache_node
                node_num += 1
    # Initialize all caches.
    if max_cache_size is None:
        max_cache_size = math.inf
    num_cached_nodes = node_num
    cache_dict_caches = {}
    cache_dict_max_lens = {}
    for i in range(num_cached_nodes):
        cache_dict_caches[i] = OrderedDict()
        cache_dict_max_lens[i] = max_cache_size / num_cached_nodes
    willump_cache_dict[WILLUMP_CACHE_NAME] = cache_dict_caches
    willump_cache_dict[WILLUMP_CACHE_MAX_LEN_NAME] = cache_dict_max_lens
    willump_cache_dict[WILLUMP_CACHE_ITER_NUMBER] = 0
    return sorted_nodes
