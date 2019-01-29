from typing import MutableMapping, List, Set, Tuple, Mapping, Optional
import typing
import ast
import copy

from weld.types import *
from willump import *

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
                # Set that node's output's new type.
                typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
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
                    input_node.push_model("linear", (model_node.weights_data_name,), selection_map)
                    typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
                    nodes_to_sum.append(input_node)
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
                # Set that node's output's new type.
                typing_map[input_node.get_output_name()] = WeldVec(WeldDouble())
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
    combiner_input_names = set()
    for model_input_node in input_nodes:
        combiner_input_names.add(model_input_node.get_output_name())
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
            async_node_index = pyblock.index(async_node)
            input_names: List[str] = list(map(lambda x: x.get_output_name(), async_node.get_in_nodes()))
            first_legal_index: int = async_node_index
            for j in range(async_node_index - 1, 0 - 1, -1):
                if pyblock[j].get_output_name() in input_names:
                    break
                else:
                    first_legal_index = j
            pyblock.remove(async_node)
            async_node_ast: ast.Assign = async_node.get_python()
            assert(isinstance(async_node_ast.value, ast.Call))
            assert(isinstance(async_node_ast.value.func, ast.Name))
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
            executor_async_node = WillumpPythonNode(async_node_ast, async_node.get_output_name(), async_node.get_in_nodes())
            pyblock.insert(first_legal_index, executor_async_node)
            last_legal_index = first_legal_index + 1
            for j in range(first_legal_index + 1, len(pyblock)):
                curr_node = pyblock[j]
                node_inputs = list(map(lambda x: x.get_output_name(), curr_node.get_in_nodes()))
                if async_node.get_output_name() in node_inputs:
                    break
                else:
                    last_legal_index = j
            result_python = "%s = %s.result()" % (async_node.get_output_name(), async_node.get_output_name())
            result_ast: ast.Module = \
                ast.parse(result_python, "exec")
            pandas_input_node = WillumpPythonNode(result_ast.body[0], async_node.get_output_name(), [executor_async_node])
            pyblock.insert(last_legal_index, pandas_input_node)
        sorted_nodes = sorted_nodes[:pyblock_start] + pyblock + sorted_nodes[pyblock_end:]
        pyblock_start_end = find_pyblock_after_n(pyblock_end)

    return sorted_nodes


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
    # Split Weld blocks into multiple threads.
    num_threads = num_workers + 1  # The main thread also does work.
    if num_threads > 1:
        threaded_statements_list = \
            multithreading_weld_blocks_pass(weld_block_node_list,
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


def graph_to_weld(graph: WillumpGraph, typing_map: Mapping[str, WeldType], batch: bool = True, num_workers=0) -> \
        List[typing.Union[ast.AST, Tuple[List[str], List[str], List[List[str]]]]]:
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
    sorted_nodes = async_python_functions_parallel_pass(sorted_nodes)
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
