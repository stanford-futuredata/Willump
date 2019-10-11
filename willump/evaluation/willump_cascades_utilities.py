import ast
import copy
import random
from typing import MutableMapping, List, Tuple, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import willump.evaluation.willump_graph_passes as wg_passes
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from willump.graph.hash_join_node import WillumpHashJoinNode
from willump.graph.identity_node import IdentityNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.pandas_column_selection_node_python import PandasColumnSelectionNodePython
from willump.graph.pandas_dataframe_concatenation_node import PandasDataframeConcatenationNode
from willump.graph.pandas_series_concatenation_node import PandasSeriesConcatenationNode
from willump.graph.pandas_to_dense_matrix_node import PandasToDenseMatrixNode
from willump.graph.reshape_node import ReshapeNode
from willump.graph.stack_dense_node import StackDenseNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_model_node import WillumpModelNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.willump_utilities import *


def graph_from_input_sources(node: WillumpGraphNode, selected_input_sources: List[WillumpGraphNode],
                             typing_map: MutableMapping[str, WeldType],
                             base_discovery_dict, which: str, small_model_output_node=None) \
        -> Optional[WillumpGraphNode]:
    """
    Take in a node and a list of input sources.  Return a node that only depends on the intersection of its
    original input sources and the list of input sources.  Return None if a node depends on no node in the list.

    This function should not modify the original graph or any of its nodes.

    It does modify the type map.
    """

    def graph_from_input_sources_recursive(node):
        return_node = None
        if isinstance(node, ArrayCountVectorizerNode) or isinstance(node, ArrayTfIdfNode):
            if node in selected_input_sources:
                if small_model_output_node is not None:
                    node.push_cascade(small_model_output_node)
                return_node = node
        elif isinstance(node, StackSparseNode):
            node_input_nodes = node.get_in_nodes()
            node_input_names = node.get_in_names()
            node_output_name = node.get_output_name()
            new_output_name = ("cascading__%s__" % which) + node_output_name
            node_output_type = node.get_output_type()
            new_input_nodes, new_input_names = [], []
            for node_input_node, node_input_name in zip(node_input_nodes, node_input_names):
                return_node = graph_from_input_sources_recursive(node_input_node)
                if return_node is not None:
                    new_input_nodes.append(return_node)
                    new_input_names.append(return_node.get_output_names()[0])
            return_node = StackSparseNode(input_nodes=new_input_nodes,
                                          input_names=new_input_names,
                                          output_name=new_output_name,
                                          output_type=node_output_type)
            typing_map[new_output_name] = node_output_type
        elif isinstance(node, StackDenseNode):
            node_input_nodes = node.get_in_nodes()
            node_input_names = node.get_in_names()
            node_output_name = node.get_output_name()
            new_output_name = ("cascading__%s__" % which) + node_output_name
            node_output_type = node.get_output_type()
            new_input_nodes, new_input_names = [], []
            for node_input_node, node_input_name in zip(node_input_nodes, node_input_names):
                return_node = graph_from_input_sources_recursive(node_input_node)
                if return_node is not None:
                    new_input_nodes.append(return_node)
                    new_input_names.append(return_node.get_output_names()[0])
            return_node = StackDenseNode(input_nodes=new_input_nodes,
                                         input_names=new_input_names,
                                         output_name=new_output_name,
                                         output_type=node_output_type)
            typing_map[new_output_name] = node_output_type
        elif isinstance(node, PandasColumnSelectionNode) or isinstance(node, PandasColumnSelectionNodePython):
            node_input_nodes = node.get_in_nodes()
            new_input_nodes = [graph_from_input_sources_recursive(node_input_node) for
                               node_input_node in node_input_nodes]
            if not all(new_input_node is None for new_input_node in new_input_nodes):
                assert all(len(new_input_node.get_output_names()) == 1 for new_input_node in new_input_nodes)
                new_input_names = [new_input_node.get_output_name() for new_input_node in new_input_nodes]
                node_output_name = node.get_output_name()
                new_output_name = ("cascading__%s__" % which) + node_output_name
                new_input_types: List[WeldType] = [new_input_node.get_output_type() for new_input_node in
                                                   new_input_nodes]
                selected_columns = node.selected_columns
                new_selected_columns = list(
                    filter(lambda x: any(x in new_input_type.column_names for new_input_type in new_input_types),
                           selected_columns))
                orig_output_type = node.get_output_type()
                if isinstance(orig_output_type, WeldPandas):
                    # Hacky code so that the pre-joins dataframe columns don't appear twice in the full output.
                    try:
                        base_node = wg_passes.find_dataframe_base_node(node_input_nodes[0], base_discovery_dict)
                        assert (isinstance(base_node, WillumpHashJoinNode))
                        if base_node.get_in_nodes()[0] not in selected_input_sources:
                            base_left_df_col_names = base_node.left_df_type.column_names
                            new_selected_columns = list(
                                filter(lambda x: x not in base_left_df_col_names, new_selected_columns))
                        col_map = {col_name: col_type for col_name, col_type in
                                   zip(orig_output_type.column_names, orig_output_type.field_types)}
                        new_field_types = [col_map[col_name] for col_name in new_selected_columns]
                    except AssertionError:
                        col_map = {col_name: col_type for col_name, col_type in
                                   zip(orig_output_type.column_names, orig_output_type.field_types)}
                        new_field_types = [col_map[col_name] for col_name in new_selected_columns]
                    new_output_type = WeldPandas(field_types=new_field_types, column_names=new_selected_columns)
                else:
                    assert (isinstance(orig_output_type, WeldSeriesPandas))
                    new_output_type = WeldSeriesPandas(elemType=orig_output_type.elemType,
                                                       column_names=new_selected_columns)
                if isinstance(node, PandasColumnSelectionNode):
                    return_node = PandasColumnSelectionNode(input_nodes=new_input_nodes,
                                                            input_names=new_input_names,
                                                            output_name=new_output_name,
                                                            input_types=new_input_types,
                                                            selected_columns=new_selected_columns,
                                                            output_type=new_output_type)
                else:
                    return_node = PandasColumnSelectionNodePython(input_nodes=new_input_nodes,
                                                                  input_names=new_input_names,
                                                                  output_name=new_output_name,
                                                                  input_types=new_input_types,
                                                                  selected_columns=new_selected_columns,
                                                                  output_type=new_output_type)
                typing_map[new_output_name] = return_node.get_output_type()
        elif isinstance(node, WillumpHashJoinNode):
            base_node = wg_passes.find_dataframe_base_node(node, base_discovery_dict)
            node_input_node = node.get_in_nodes()[0]
            if node in selected_input_sources:
                if node is base_node:
                    new_input_node = base_node.get_in_nodes()[0]
                    new_input_name = base_node.get_in_names()[0]
                    new_input_type = base_node.left_df_type
                else:
                    new_input_node = graph_from_input_sources_recursive(node_input_node)
                    if new_input_node is None:
                        new_input_node = base_node.get_in_nodes()[0]
                        new_input_name = base_node.get_in_names()[0]
                        new_input_type = base_node.left_df_type
                    else:
                        assert (isinstance(new_input_node, WillumpHashJoinNode))
                        new_input_name = new_input_node.get_output_name()
                        new_input_type = new_input_node.get_output_type()
                return_node = copy.copy(node)
                if small_model_output_node is not None:
                    return_node.push_cascade(small_model_output_node)
                return_node._input_nodes = copy.copy(node._input_nodes)
                return_node._input_nodes[0] = new_input_node
                return_node._input_names = copy.copy(node._input_names)
                return_node._input_names[0] = new_input_name
                return_node.left_input_name = new_input_name
                return_node.left_df_type = new_input_type
                return_node._output_type = \
                    WeldPandas(field_types=new_input_type.field_types + node.right_df_type.field_types,
                               column_names=new_input_type.column_names + node.right_df_type.column_names)
                typing_map[return_node.get_output_name()] = return_node.get_output_type()
            elif node is not base_node:
                return graph_from_input_sources_recursive(node_input_node)
            else:
                pass
        elif isinstance(node, PandasSeriesConcatenationNode):
            node_input_nodes = node.get_in_nodes()
            node_input_names = node.get_in_names()
            node_output_name = node.get_output_name()
            new_output_name = ("cascading__%s__" % which) + node_output_name
            node_output_type: WeldSeriesPandas = node.get_output_type()
            new_input_nodes, new_input_names, new_input_types, new_output_columns = [], [], [], []
            for node_input_node, node_input_name in zip(node_input_nodes, node_input_names):
                return_node = graph_from_input_sources_recursive(node_input_node)
                if return_node is not None:
                    new_input_nodes.append(return_node)
                    new_input_names.append(node_input_name)
                    return_node_output_type = return_node.get_output_types()[0]
                    assert (isinstance(return_node_output_type, WeldSeriesPandas))
                    new_input_types.append(return_node_output_type)
                    new_output_columns += return_node_output_type.column_names
            new_output_type = WeldSeriesPandas(elemType=node_output_type.elemType, column_names=new_output_columns)
            return_node = PandasSeriesConcatenationNode(input_nodes=new_input_nodes,
                                                        input_names=new_input_names,
                                                        input_types=new_input_types,
                                                        output_type=new_output_type,
                                                        output_name=new_output_name)
            typing_map[new_output_name] = return_node.get_output_type()
        elif isinstance(node, PandasDataframeConcatenationNode):
            node_input_nodes = node.get_in_nodes()
            node_input_names = node.get_in_names()
            node_output_name = node.get_output_name()
            new_output_name = ("cascading__%s__" % which) + node_output_name
            new_input_nodes, new_input_names, new_input_types, new_output_columns, new_field_types \
                = [], [], [], [], []
            for node_input_node, node_input_name in zip(node_input_nodes, node_input_names):
                return_node = graph_from_input_sources_recursive(node_input_node)
                if return_node is not None:
                    new_input_nodes.append(return_node)
                    new_input_names.append(node_input_name)
                    return_node_output_type = return_node.get_output_types()[0]
                    assert (isinstance(return_node_output_type, WeldPandas))
                    new_input_types.append(return_node_output_type)
                    new_output_columns += return_node_output_type.column_names
                    new_field_types += return_node_output_type.field_types
            new_output_type = WeldPandas(field_types=new_field_types, column_names=new_output_columns)
            return_node = PandasDataframeConcatenationNode(input_nodes=new_input_nodes,
                                                           input_names=new_input_names,
                                                           input_types=new_input_types,
                                                           output_type=new_output_type,
                                                           output_name=new_output_name,
                                                           keyword_args=node.keyword_args)
            typing_map[new_output_name] = return_node.get_output_type()
        elif isinstance(node, IdentityNode):
            input_node = node.get_in_nodes()[0]
            new_input_node = graph_from_input_sources_recursive(input_node)
            if new_input_node is not None:
                output_name = node.get_output_name()
                new_output_name = ("cascading__%s__" % which) + output_name
                output_type = new_input_node.get_output_type()
                new_input_name = new_input_node.get_output_name()
                return_node = IdentityNode(input_node=new_input_node, input_name=new_input_name,
                                           output_name=new_output_name, output_type=output_type)
                typing_map[new_output_name] = output_type
        elif isinstance(node, ReshapeNode):
            input_node = node.get_in_nodes()[0]
            new_input_node = graph_from_input_sources_recursive(input_node)
            if new_input_node is not None:
                output_name = node.get_output_name()
                new_output_name = ("cascading__%s__" % which) + output_name
                output_type = node.get_output_type()
                new_input_name = new_input_node.get_output_name()
                return_node = ReshapeNode(input_node=new_input_node, input_name=new_input_name,
                                          output_name=new_output_name, output_type=output_type,
                                          reshape_args=node.reshape_args)
                typing_map[new_output_name] = output_type
        elif isinstance(node, PandasToDenseMatrixNode):
            input_node = node.get_in_nodes()[0]
            new_input_node = graph_from_input_sources_recursive(input_node)
            if new_input_node is not None:
                output_name = node.get_output_name()
                new_output_name = ("cascading__%s__" % which) + output_name
                output_type = node.get_output_type()
                new_input_name = new_input_node.get_output_name()
                return_node = PandasToDenseMatrixNode(input_node=new_input_node, input_name=new_input_name,
                                                      input_type=new_input_node.get_output_type(),
                                                      output_name=new_output_name, output_type=output_type)
                typing_map[new_output_name] = output_type
        elif isinstance(node, WillumpPythonNode):
            if node in selected_input_sources:
                if small_model_output_node is not None:
                    pass
                return_node = node
        elif isinstance(node, WillumpInputNode):
            if node in selected_input_sources:
                return_node = node
        else:
            panic("Unrecognized node found when making cascade %s" % node.__repr__())
        return return_node

    return graph_from_input_sources_recursive(node)


def get_model_node_dependencies(training_input_node: WillumpGraphNode, base_discovery_dict, typing_map,
                                small_model_output_node=None) \
        -> List[WillumpGraphNode]:
    """
    Take in a training node's input.  Return a Weld block that constructs the training node's
    input from its input sources.
    """
    current_node_stack: List[WillumpGraphNode] = [training_input_node]
    # Nodes through which the model has been pushed which are providing output.
    output_block: List[WillumpGraphNode] = []
    shortening_nodes = {}
    while len(current_node_stack) > 0:
        input_node = current_node_stack.pop()
        if isinstance(input_node, ArrayCountVectorizerNode) or isinstance(input_node, ArrayTfIdfNode):
            output_block.insert(0, input_node)
        elif isinstance(input_node, StackSparseNode) or isinstance(input_node, PandasColumnSelectionNode) \
                or isinstance(input_node, PandasSeriesConcatenationNode) or isinstance(input_node, IdentityNode) \
                or isinstance(input_node, ReshapeNode) or isinstance(input_node, PandasToDenseMatrixNode) \
                or isinstance(input_node, WillumpInputNode) \
                or isinstance(input_node, PandasDataframeConcatenationNode) \
                or isinstance(input_node, PandasColumnSelectionNodePython) \
                or isinstance(input_node, StackDenseNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, WillumpHashJoinNode):
            base_node = wg_passes.find_dataframe_base_node(input_node, base_discovery_dict)
            output_block.insert(0, input_node)
            if input_node is not base_node:
                join_left_input_node = input_node.get_in_nodes()[0]
                current_node_stack.append(join_left_input_node)
        elif isinstance(input_node, WillumpPythonNode):
            output_block.insert(0, input_node)
            node_output_types = input_node.get_output_types()
            # If the node is cascaded, only execute it on low-confidence data inputs.
            if small_model_output_node is not None and \
                    len(node_output_types) == 1 and ((isinstance(node_output_types[0], WeldPandas)
                                                      and len(input_node.get_in_nodes()) == 1)
                                                     or (isinstance(node_output_types[0], WeldVec) and
                                                         node_output_types[0].width is not None)
                                                     or isinstance(node_output_types[0], WeldCSR)):
                small_model_output_name = strip_linenos_from_var(small_model_output_node.get_output_names()[0])
                for i, node_input_name in enumerate(input_node.get_in_names()):
                    input_type = typing_map[node_input_name]
                    # Only shorten data.
                    if (isinstance(input_type, WeldPandas) or (isinstance(input_type, WeldVec) and
                                                               input_type.width is not None)
                            or isinstance(input_type, WeldCSR)):
                        node_input_name = strip_linenos_from_var(node_input_name)
                        shorten_python_code = "%s = cascade_df_shorten(%s, %s)" % (node_input_name,
                                                                                   node_input_name,
                                                                                   small_model_output_name)
                        shorten_python_ast: ast.Module = \
                            ast.parse(shorten_python_code, "exec")
                        shorten_python_node = WillumpPythonNode(python_ast=shorten_python_ast.body[0],
                                                                input_names=[input_node.get_in_names()[0],
                                                                             small_model_output_node.get_output_names()[
                                                                                 0]],
                                                                output_names=[input_node.get_in_names()[0]],
                                                                output_types=input_node.get_in_nodes()[
                                                                    0].get_output_types(),
                                                                in_nodes=[input_node, small_model_output_node])
                        shortening_nodes[node_input_name] = shorten_python_node
                        input_node._in_nodes[i] = shorten_python_node
        else:
            panic("Unrecognized node found when making cascade dependencies: %s" % input_node.__repr__())
    for shortening_node in shortening_nodes.values():
        output_block.insert(0, shortening_node)
    return output_block


def create_indices_to_costs_map(training_node: WillumpModelNode) -> Mapping[tuple, float]:
    """
    Create a map from the indices of the features generated by an operator (used as a unique identifier of the
    operator) to the operator's cost.
    """
    training_node_inputs: Mapping[
        WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]] = training_node.get_model_inputs()
    indices_to_costs_map: MutableMapping = {}
    for node, indices in training_node_inputs.items():
        if isinstance(indices, tuple):
            pass
        else:
            indices = tuple(indices.values())
        indices_to_costs_map[indices] = node.get_cost()
    return indices_to_costs_map


def split_model_inputs(model_node: WillumpModelNode, feature_importances, indices_to_costs_map,
                       cost_cutoff) -> \
        Tuple[List[WillumpGraphNode], List[WillumpGraphNode], List[int], List[int], int, int]:
    """
    Use a model's feature importances to divide its inputs into those more and those less important.  Return
    lists of each.
    """

    def knapsack_dp(values, weights, capacity):
        # Credit: https://gist.github.com/KaiyangZhou/71a473b1561e0ea64f97d0132fe07736
        n_items = len(values)
        table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
        keep = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)

        for i in range(1, n_items + 1):
            for w in range(0, capacity + 1):
                wi = weights[i - 1]  # weight of current item
                vi = values[i - 1]  # value of current item
                if (wi <= w) and (vi + table[i - 1, w - wi] > table[i - 1, w]):
                    table[i, w] = vi + table[i - 1, w - wi]
                    keep[i, w] = 1
                else:
                    table[i, w] = table[i - 1, w]
        picks = []
        K = capacity
        for i in range(n_items, 0, -1):
            if keep[i, K] == 1:
                picks.append(i)
                K -= weights[i - 1]
        picks.sort()
        picks = [x - 1 for x in picks]  # change to 0-index
        return picks

    training_node_inputs: Mapping[
        WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]] = model_node.get_model_inputs()
    nodes_to_importances: MutableMapping[WillumpGraphNode, float] = {}
    nodes_to_indices: MutableMapping[WillumpGraphNode, tuple] = {}
    nodes_to_costs: MutableMapping[WillumpGraphNode, int] = {}
    orig_total_cost = sum(indices_to_costs_map.values())
    scaled_total_cost = 1000
    scale_factor = scaled_total_cost / orig_total_cost
    for node, indices in training_node_inputs.items():
        if isinstance(indices, tuple):
            node_importance = feature_importances[indices]
            start, end = indices
            nodes_to_indices[node] = tuple(range(start, end))
        else:
            indices = tuple(indices.values())
            node_importance = feature_importances[indices]
            nodes_to_indices[node] = indices
        nodes_to_importances[node] = node_importance
        node_cost: int = round(indices_to_costs_map[indices] * scale_factor)
        nodes_to_costs[node] = node_cost
    assert (scaled_total_cost - len(training_node_inputs) <= sum(nodes_to_costs.values()) <= scaled_total_cost + len(
        training_node_inputs))
    nodes, importances, costs = list(nodes_to_indices.keys()), [], []
    for node in nodes:
        importances.append(nodes_to_importances[node])
        costs.append(nodes_to_costs[node])
    picks_indices = knapsack_dp(importances, costs, int(scaled_total_cost * cost_cutoff))
    more_important_inputs = [nodes[i] for i in picks_indices]
    current_cost: int = sum([costs[i] for i in picks_indices])
    mi_feature_indices: List[int] = []
    li_feature_indices: List[int] = []
    for node in more_important_inputs:
        for i in nodes_to_indices[node]:
            mi_feature_indices.append(i)
    less_important_inputs = [entry for entry in nodes if entry not in more_important_inputs]
    for node in less_important_inputs:
        for i in nodes_to_indices[node]:
            li_feature_indices.append(i)
    return more_important_inputs, less_important_inputs, mi_feature_indices, li_feature_indices, current_cost, scaled_total_cost


def calculate_feature_importance(valid_x, valid_y, orig_model, train_predict_score_functions: tuple, model_inputs) -> \
        Tuple[Mapping[tuple, float], object]:
    """
    Calculate the importance of all operators' feature sets using mean decrease accuracy.
    Return a map from the indices of the features generated by an operator (used as a unique identifier of the
    operator) to the operator's importance.
    """
    willump_train_function, willump_predict_function, _, willump_score_function = train_predict_score_functions
    base_preds = willump_predict_function(orig_model, valid_x)
    base_score = willump_score_function(valid_y, base_preds)
    return_map = {}
    for node, indices in model_inputs.items():
        valid_x_copy = valid_x.copy()
        if isinstance(valid_x_copy, pd.DataFrame):
            valid_x_copy = valid_x_copy.values
        shuffle_order = np.arange(valid_x_copy.shape[0])
        np.random.shuffle(shuffle_order)
        if isinstance(indices, tuple):
            start, end = indices
            for i in range(0, valid_x.shape[0], 20000):
                valid_x_copy[i:i + 1000, start:end] = valid_x[shuffle_order[i:i + 1000], start:end]
        else:
            indices = tuple(indices.values())
            for i in indices:
                valid_x_copy[:, i] = valid_x_copy[shuffle_order, i]
        shuffled_preds = willump_predict_function(orig_model, valid_x_copy)
        shuffled_score = willump_score_function(valid_y, shuffled_preds)
        return_map[indices] = base_score - shuffled_score
        del valid_x_copy
    return return_map, orig_model


def calculate_feature_set_performance(train_x, train_y, valid_x, valid_y, train_predict_score_functions: tuple,
                                      mi_feature_indices: List[int],
                                      orig_model, mi_cost: float, total_cost: float):
    willump_train_function, willump_predict_function, willump_predict_proba_function, willump_score_function = \
        train_predict_score_functions
    train_x_efficient, valid_x_efficient = train_x[:, mi_feature_indices], valid_x[:, mi_feature_indices]
    small_model = willump_train_function(train_x_efficient, train_y)
    small_confidences = willump_predict_proba_function(small_model, valid_x_efficient)
    small_preds = willump_predict_function(small_model, valid_x_efficient)
    orig_preds = willump_predict_function(orig_model, valid_x)
    orig_score = willump_score_function(valid_y, orig_preds)
    threshold_to_combined_cost_map = {}
    for cascade_threshold in [0.6, 0.7, 0.8, 0.9, 1.0]:
        combined_preds = orig_preds.copy()
        num_mi_predicted = 0
        for i in range(len(small_confidences)):
            if small_confidences[i] > cascade_threshold or small_confidences[i] < 1 - cascade_threshold:
                num_mi_predicted += 1
                combined_preds[i] = small_preds[i]
        combined_score = willump_score_function(valid_y, combined_preds)
        frac_mi_predicted = num_mi_predicted / len(combined_preds)
        combined_cost = frac_mi_predicted * mi_cost + (1 - frac_mi_predicted) * total_cost
        if combined_score > orig_score - 0.001:
            threshold_to_combined_cost_map[cascade_threshold] = combined_cost
    best_threshold, best_cost = min(threshold_to_combined_cost_map.items(), key=lambda x: x[1])
    return best_threshold, best_cost


def calculate_feature_set_performance_top_k(train_x, train_y, valid_x, orig_model_probs,
                                            train_predict_score_functions: tuple,
                                            mi_feature_indices: List[int],
                                            mi_cost: float, total_cost: float, top_k_distribution: List[int],
                                            valid_size_distribution: List[int]):
    min_precision = 0.95
    willump_train_function, willump_predict_function, willump_predict_proba_function, willump_score_function = \
        train_predict_score_functions
    assert (valid_x.shape[0] > max(valid_size_distribution))
    train_x_efficient, valid_x_efficient = train_x[:, mi_feature_indices], valid_x[:, mi_feature_indices]
    small_model = willump_train_function(train_x_efficient, train_y)
    small_probs = willump_predict_proba_function(small_model, valid_x_efficient)
    num_samples = 100
    candidate_ratios = sorted(list(set(range(1, 100)).union(set(range(1, min(valid_size_distribution) //
                                                                      max(top_k_distribution), 10)))))
    ratios_map = {i: 0 for i in candidate_ratios}
    for i in range(num_samples):
        valid_size = random.choice(valid_size_distribution)
        top_k = random.choice(top_k_distribution)
        sample_indices = np.random.choice(len(orig_model_probs), size=valid_size)
        sample_small_probs = small_probs[sample_indices]
        sample_orig_probs = orig_model_probs[sample_indices]
        assert (len(sample_small_probs) == len(sample_orig_probs) == valid_size)
        orig_model_top_k_idx = np.argsort(sample_orig_probs)[-1 * top_k:]
        small_model_probs_idx_sorted = np.argsort(sample_small_probs)
        overlap_count = 0
        for i in range(len(candidate_ratios)):
            ratio = candidate_ratios[i]
            if i == 0:
                small_model_top_ratio_k_idx = small_model_probs_idx_sorted[-1 * top_k * ratio:]
            else:
                prev_ratio = candidate_ratios[i - 1]
                small_model_top_ratio_k_idx = small_model_probs_idx_sorted[-1 * top_k * ratio: -1 * top_k * prev_ratio]
            overlap_count += len(np.intersect1d(orig_model_top_k_idx, small_model_top_ratio_k_idx))
            small_model_precision = overlap_count / top_k
            if small_model_precision >= min_precision:
                ratios_map[ratio] += 1
    # This implies that if the precisions are normally distributed, given n=100 samples, the probability that
    # the precision is above min_precision is 95%.
    good_ratios = [ratio for ratio in candidate_ratios if ratios_map[ratio] >= 0.99 * num_samples]
    good_ratio = min(good_ratios)
    cost = mi_cost * np.mean(valid_size_distribution) + (total_cost - mi_cost) * good_ratio * \
           np.mean(top_k_distribution)
    return good_ratio, cost
