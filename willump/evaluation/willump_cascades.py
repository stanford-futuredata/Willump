import ast
import copy
from typing import MutableMapping, List, Tuple, Mapping, Optional

import willump.evaluation.willump_graph_passes as wg_passes
from willump import *
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from willump.graph.cascade_column_selection_node import CascadeColumnSelectionNode
from willump.graph.cascade_combine_predictions_node import CascadeCombinePredictionsNode
from willump.graph.cascade_stack_dense_node import CascadeStackDenseNode
from willump.graph.cascade_stack_sparse_node import CascadeStackSparseNode
from willump.graph.cascade_threshold_proba_node import CascadeThresholdProbaNode
from willump.graph.hash_join_node import WillumpHashJoinNode
from willump.graph.identity_node import IdentityNode
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.pandas_series_concatenation_node import PandasSeriesConcatenationNode
from willump.graph.reshape_node import ReshapeNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.trees_model_node import TreesModelNode
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_model_node import WillumpModelNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.willump_training_node import WillumpTrainingNode
from willump.willump_utilities import *


def graph_from_input_sources(node: WillumpGraphNode, selected_input_sources: List[WillumpGraphNode],
                             typing_map: MutableMapping[str, WeldType],
                             base_discovery_dict, which: str, small_model_output_name=None) \
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
                if small_model_output_name is not None:
                    node.push_cascade(small_model_output_name)
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
                    new_input_names.append(return_node.get_output_name())
            return_node = StackSparseNode(input_nodes=new_input_nodes,
                                          input_names=new_input_names,
                                          output_name=new_output_name,
                                          output_type=node_output_type)
            typing_map[new_output_name] = node_output_type
        elif isinstance(node, PandasColumnSelectionNode):
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
                    col_map = {col_name: col_type for col_name, col_type in
                               zip(orig_output_type.column_names, orig_output_type.field_types)}
                    new_field_types = [col_map[col_name] for col_name in new_selected_columns]
                    new_output_type = WeldPandas(field_types=new_field_types, column_names=new_selected_columns)
                else:
                    assert (isinstance(orig_output_type, WeldSeriesPandas))
                    new_output_type = WeldSeriesPandas(elemType=orig_output_type.elemType,
                                                       column_names=new_selected_columns)
                return_node = PandasColumnSelectionNode(input_nodes=new_input_nodes,
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
                if small_model_output_name is not None:
                    return_node.push_cascade(small_model_output_name)
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
        elif isinstance(node, WillumpPythonNode):
            if node in selected_input_sources:
                if small_model_output_name is not None:
                    pass
                return_node = node
        else:
            panic("Unrecognized node found when making cascade %s" % node.__repr__())
        return return_node

    return graph_from_input_sources_recursive(node)


def get_model_node_dependencies(training_input_node: WillumpGraphNode, base_discovery_dict) \
        -> List[WillumpGraphNode]:
    """
    Take in a training node's input.  Return a Weld block that constructs the training node's
    input from its input sources.
    """
    current_node_stack: List[WillumpGraphNode] = [training_input_node]
    # Nodes through which the model has been pushed which are providing output.
    output_block: List[WillumpGraphNode] = []
    while len(current_node_stack) > 0:
        input_node = current_node_stack.pop()
        if isinstance(input_node, ArrayCountVectorizerNode) or isinstance(input_node, ArrayTfIdfNode):
            output_block.insert(0, input_node)
        elif isinstance(input_node, StackSparseNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, PandasColumnSelectionNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, PandasSeriesConcatenationNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, WillumpHashJoinNode):
            base_node = wg_passes.find_dataframe_base_node(input_node, base_discovery_dict)
            output_block.insert(0, input_node)
            if input_node is not base_node:
                join_left_input_node = input_node.get_in_nodes()[0]
                current_node_stack.append(join_left_input_node)
        elif isinstance(input_node, IdentityNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, ReshapeNode):
            output_block.insert(0, input_node)
            current_node_stack += input_node.get_in_nodes()
        elif isinstance(input_node, WillumpPythonNode):
            output_block.insert(0, input_node)
            if input_node.does_not_modify_data:
                current_node_stack += input_node.get_in_nodes()
        else:
            panic("Unrecognized node found when making cascade dependencies: %s" % input_node.__repr__())
    return output_block


def get_combiner_node(node_one: WillumpGraphNode, node_two: WillumpGraphNode, orig_node: WillumpGraphNode) \
        -> WillumpGraphNode:
    """
    Generate a node that will fuse node_one and node_two to match the output of orig_node.  Requires all
    three nodes be of the same type.
    """
    assert (len(orig_node.get_output_types()) == len(node_one.get_output_types()) == len(
        node_two.get_output_types()) == 1)
    orig_output_type = orig_node.get_output_types()[0]
    if isinstance(orig_output_type, WeldCSR):
        return StackSparseNode(input_nodes=[node_one, node_two],
                               input_names=[node_one.get_output_names()[0], node_two.get_output_names()[0]],
                               output_name=orig_node.get_output_names()[0],
                               output_type=orig_output_type)
    elif isinstance(orig_output_type, WeldPandas):
        return PandasColumnSelectionNode(input_nodes=[node_one, node_two],
                                         input_names=[node_one.get_output_names()[0], node_two.get_output_names()[0]],
                                         output_name=orig_node.get_output_names()[0],
                                         input_types=[node_one.get_output_types()[0], node_two.get_output_types()[0]],
                                         selected_columns=orig_output_type.column_names,
                                         output_type=orig_output_type)
    else:
        panic("Unrecognized combiner output type %s" % orig_output_type)


def split_model_inputs(model_node: WillumpModelNode, feature_importances) -> \
        Tuple[List[WillumpGraphNode], List[WillumpGraphNode]]:
    """
    Use a model's feature importances to divide its inputs into those more and those less important.  Return
    lists of each.
    """
    training_node_inputs: Mapping[
        WillumpGraphNode, Union[Tuple[int, int], Mapping[str, int]]] = model_node.get_model_inputs()
    nodes_to_importances: MutableMapping[WillumpGraphNode, float] = {}
    for node, indices in training_node_inputs.items():
        if isinstance(indices, tuple):
            start, end = indices
            sum_importance = sum(feature_importances[start:end])
        else:
            indices = indices.values()
            sum_importance = sum([feature_importances[i] for i in indices])
        nodes_to_importances[node] = sum_importance
    ranked_inputs = sorted(nodes_to_importances.keys(), key=lambda x: nodes_to_importances[x])
    more_important_inputs = ranked_inputs[len(ranked_inputs) // 2:]
    less_important_inputs = ranked_inputs[:len(ranked_inputs) // 2]
    return more_important_inputs, less_important_inputs


def training_model_cascade_pass(sorted_nodes: List[WillumpGraphNode],
                                typing_map: MutableMapping[str, WeldType],
                                training_cascades: dict) -> List[WillumpGraphNode]:
    """
    Take in a program training a model.  Rank features in the model by importance.  Partition
    features into "more important" and "less important."  Train a small model on only more important features
    and a big model on all features.
    """

    def recreate_training_node(new_input_node: WillumpGraphNode, orig_node: WillumpTrainingNode,
                               output_prefix) -> WillumpTrainingNode:
        """
        Create a node based on orig_node that uses new_input_node as its input and prefixes its output's name
        with output_prefix.
        """
        new_input_nodes = copy.copy(orig_node.get_in_nodes())
        new_input_nodes[0] = new_input_node
        new_input_names = copy.copy(orig_node.get_in_names())
        new_input_names[0] = new_input_node.get_output_name()
        node_output_name = orig_node.get_output_names()[0]
        new_output_name = output_prefix + node_output_name
        node_feature_importances = orig_node.get_feature_importances()
        new_python_ast = copy.deepcopy(orig_node.get_python())
        new_python_ast.value.args[0].id = strip_linenos_from_var(new_input_node.get_output_name())
        new_python_ast.value.func.value.id = strip_linenos_from_var(new_output_name)
        new_python_ast.targets[0].id = strip_linenos_from_var(new_output_name)
        return WillumpTrainingNode(python_ast=new_python_ast, input_names=new_input_names, in_nodes=new_input_nodes,
                                   output_names=[new_output_name], feature_importances=node_feature_importances)

    for node in sorted_nodes:
        if isinstance(node, WillumpTrainingNode):
            training_node: WillumpTrainingNode = node
            break
    else:
        return sorted_nodes
    feature_importances = training_node.get_feature_importances()
    training_cascades["feature_importances"] = feature_importances
    more_important_inputs, less_important_inputs = split_model_inputs(training_node, feature_importances)
    training_input_node = training_node.get_in_nodes()[0]
    # Create Willump graphs and code blocks that produce the more and less important inputs.
    base_discovery_dict = {}
    less_important_inputs_head = graph_from_input_sources(training_input_node, less_important_inputs, typing_map,
                                                          base_discovery_dict, "less")
    less_important_inputs_block = get_model_node_dependencies(less_important_inputs_head, base_discovery_dict)
    base_discovery_dict = {}
    more_important_inputs_head = graph_from_input_sources(training_input_node, more_important_inputs, typing_map,
                                                          base_discovery_dict, "more")
    more_important_inputs_block = get_model_node_dependencies(more_important_inputs_head, base_discovery_dict)
    combiner_node = get_combiner_node(more_important_inputs_head, less_important_inputs_head, training_input_node)
    small_training_node = recreate_training_node(more_important_inputs_head, training_node, "small_")
    big_training_node = recreate_training_node(combiner_node, training_node, "")
    # Store the big model for evaluation.
    big_model_python_name = strip_linenos_from_var(big_training_node.get_output_names()[0])
    add_big_model_python = "%s[\"big_model\"] = %s" % (WILLUMP_TRAINING_CASCADE_NAME, big_model_python_name)
    add_big_model_ast: ast.Module = \
        ast.parse(add_big_model_python, "exec")
    add_big_model_node = WillumpPythonNode(python_ast=add_big_model_ast.body[0], input_names=[big_model_python_name],
                                           output_names=[], output_types=[], in_nodes=[big_training_node])
    # Store the small model for evaluation.
    small_model_python_name = strip_linenos_from_var(small_training_node.get_output_names()[0])
    add_small_model_python = "%s[\"small_model\"] = %s" % (WILLUMP_TRAINING_CASCADE_NAME, small_model_python_name)
    add_small_model_ast: ast.Module = \
        ast.parse(add_small_model_python, "exec")
    add_small_model_node = WillumpPythonNode(python_ast=add_small_model_ast.body[0],
                                             input_names=[small_model_python_name],
                                             output_names=[], output_types=[], in_nodes=[small_training_node])
    # Copy the original model so the small and big models aren't the same.
    duplicate_model_python = "%s = copy.copy(%s)" % (small_model_python_name, big_model_python_name)
    duplicate_model_ast: ast.Module = \
        ast.parse(duplicate_model_python, "exec")
    duplicate_model_node = WillumpPythonNode(python_ast=duplicate_model_ast.body[0],
                                             input_names=[big_model_python_name],
                                             output_names=[], output_types=[], in_nodes=[big_training_node])
    base_discovery_dict = {}
    # Remove the original code for creating model inputs to replace with the new code.
    training_dependencies = get_model_node_dependencies(training_input_node, base_discovery_dict)
    for node in training_dependencies:
        sorted_nodes.remove(node)
    # Add all the new code for creating model inputs and training from them.
    training_node_index = sorted_nodes.index(training_node)
    sorted_nodes = sorted_nodes[:training_node_index] + more_important_inputs_block + less_important_inputs_block \
                   + [combiner_node, big_training_node, duplicate_model_node, small_training_node,
                      add_big_model_node, add_small_model_node] \
                   + sorted_nodes[training_node_index + 1:]
    return sorted_nodes


def eval_model_cascade_pass(sorted_nodes: List[WillumpGraphNode],
                            typing_map: MutableMapping[str, WeldType],
                            aux_data: List[Tuple[int, WeldType]],
                            eval_cascades: dict,
                            cascade_threshold: float) -> List[WillumpGraphNode]:
    """
    Take in a program with a model.  Use pre-computed feature importances to partition the models' input
    sources into those more and less important.  Rewrite the program so it first evaluates a pre-trained smaller
    model on all more-important input sources and checks confidence.  If it is above some threshold, use
    that prediction, otherwise, predict with a pre-trained bigger model on all inputs.
    """

    def get_small_model_nodes(orig_model_node: WillumpModelNode, new_input_node: WillumpGraphNode, new_model, aux_data) \
            -> Tuple[WillumpModelNode, CascadeThresholdProbaNode]:
        proba_output_name = "small__proba_" + orig_model_node.get_output_name()
        output_type = WeldVec(WeldDouble())
        typing_map[proba_output_name] = output_type
        assert (len(new_input_node.get_output_names()), 1)
        new_input_name = new_input_node.get_output_names()[0]
        if isinstance(orig_model_node, LinearRegressionNode):
            predict_proba_node = LinearRegressionNode(input_node=new_input_node,
                                                      input_name=new_input_name,
                                                      input_type=typing_map[new_input_name],
                                                      output_name=proba_output_name, output_type=output_type,
                                                      logit_weights=new_model.coef_,
                                                      logit_intercept=new_model.intercept_,
                                                      aux_data=aux_data, predict_proba=True)
        elif isinstance(orig_model_node, TreesModelNode):
            predict_proba_node = TreesModelNode(input_node=new_input_node,
                                                input_name=new_input_name,
                                                output_name=proba_output_name,
                                                model_name=SMALL_MODEL_NAME,
                                                input_width=orig_model_node.input_width, predict_proba=True,
                                                output_type=output_type)
        else:
            assert False
        threshold_output_name = "small_preds_" + orig_model_node.get_output_name()
        threshold_node = CascadeThresholdProbaNode(input_node=predict_proba_node, input_name=proba_output_name,
                                                   output_name=threshold_output_name,
                                                   threshold=cascade_threshold)
        typing_map[threshold_output_name] = WeldVec(WeldChar())
        return predict_proba_node, threshold_node

    def get_big_model_nodes(orig_model_node: WillumpModelNode, new_input_node: WillumpGraphNode, new_model, aux_data,
                            small_model_output_node: CascadeThresholdProbaNode, small_model_output_name: str) \
            -> Tuple[WillumpModelNode, CascadeCombinePredictionsNode]:
        output_name = orig_model_node.get_output_name()
        output_type = orig_model_node.get_output_type()
        assert (len(new_input_node.get_output_names()), 1)
        new_input_name = new_input_node.get_output_names()[0]
        if isinstance(orig_model_node, LinearRegressionNode):
            big_model_output = LinearRegressionNode(input_node=new_input_node,
                                                    input_name=new_input_name,
                                                    input_type=typing_map[new_input_name],
                                                    output_name=output_name, output_type=output_type,
                                                    logit_weights=new_model.coef_, logit_intercept=new_model.intercept_,
                                                    aux_data=aux_data)
        elif isinstance(orig_model_node, TreesModelNode):
            big_model_output = TreesModelNode(input_node=new_input_node,
                                              input_name=new_input_name,
                                              output_name=output_name,
                                              model_name=orig_model_node.model_name,
                                              input_width=orig_model_node.input_width,
                                              output_type=output_type)
        else:
            assert False
        combining_node = CascadeCombinePredictionsNode(big_model_predictions_node=big_model_output,
                                                       big_model_predictions_name=output_name,
                                                       small_model_predictions_node=small_model_output_node,
                                                       small_model_predictions_name=small_model_output_name,
                                                       output_name=output_name,
                                                       output_type=output_type)
        return big_model_output, combining_node

    def get_combiner_node_eval(mi_head: WillumpGraphNode, li_head: WillumpGraphNode, orig_node: WillumpGraphNode,
                               small_model_output_node: CascadeThresholdProbaNode) \
            -> WillumpGraphNode:
        """
        Generate a node that will fuse node_one and node_two to match the output of orig_node.  Requires all
        three nodes be of the same type.
        """
        assert (len(orig_node.get_output_types()) == len(mi_head.get_output_types()) == len(
            li_head.get_output_types()) == 1)
        orig_output_type = orig_node.get_output_types()[0]
        if isinstance(orig_output_type, WeldCSR):
            return CascadeStackSparseNode(more_important_nodes=[mi_head],
                                          more_important_names=[mi_head.get_output_names()[0]],
                                          less_important_nodes=[li_head],
                                          less_important_names=[li_head.get_output_names()[0]],
                                          small_model_output_node=small_model_output_node,
                                          small_model_output_name=small_model_output_node.get_output_name(),
                                          output_name=orig_node.get_output_names()[0],
                                          output_type=orig_output_type)
        elif isinstance(orig_output_type, WeldPandas):
            return CascadeColumnSelectionNode(more_important_nodes=[mi_head],
                                              more_important_names=[mi_head.get_output_names()[0]],
                                              more_important_types=[mi_head.get_output_types()[0]],
                                              less_important_nodes=[li_head],
                                              less_important_names=[li_head.get_output_names()[0]],
                                              less_important_types=[li_head.get_output_types()[0]],
                                              output_name=orig_node.get_output_names()[0],
                                              small_model_output_node=small_model_output_node,
                                              small_model_output_name=small_model_output_node.get_output_name(),
                                              selected_columns=orig_output_type.column_names)
        elif isinstance(orig_output_type, WeldVec):
            assert (isinstance(orig_output_type.elemType, WeldVec))
            return CascadeStackDenseNode(more_important_nodes=[mi_head],
                                         more_important_names=[mi_head.get_output_names()[0]],
                                         less_important_nodes=[li_head],
                                         less_important_names=[li_head.get_output_names()[0]],
                                         small_model_output_node=small_model_output_node,
                                         small_model_output_name=small_model_output_node.get_output_name(),
                                         output_name=orig_node.get_output_names()[0],
                                         output_type=orig_output_type)
        else:
            panic("Unrecognized eval combiner output type %s" % orig_output_type)

    for node in sorted_nodes:
        if isinstance(node, WillumpModelNode):
            model_node: WillumpModelNode = node
            break
    else:
        return sorted_nodes
    feature_importances = eval_cascades["feature_importances"]
    big_model = eval_cascades["big_model"]
    small_model = eval_cascades["small_model"]
    more_important_inputs, less_important_inputs = split_model_inputs(model_node, feature_importances)
    model_input_node = model_node.get_in_nodes()[0]
    # Create Willump graphs and code blocks that produce the more important inputs.
    base_discovery_dict = {}
    more_important_inputs_head = graph_from_input_sources(model_input_node, more_important_inputs, typing_map,
                                                          base_discovery_dict, "more")
    more_important_inputs_block = get_model_node_dependencies(more_important_inputs_head, base_discovery_dict)
    # The small model predicts all examples from the more important inputs.
    new_small_model_node, threshold_node = \
        get_small_model_nodes(model_node, more_important_inputs_head, small_model, aux_data)
    small_model_preds_name = threshold_node.get_output_name()
    # Less important inputs are materialized only if the small model lacks confidence in an example.
    base_discovery_dict = {}
    less_important_inputs_head = graph_from_input_sources(model_input_node, less_important_inputs, typing_map,
                                                          base_discovery_dict, "less",
                                                          small_model_output_name=small_model_preds_name)
    less_important_inputs_block = get_model_node_dependencies(less_important_inputs_head, base_discovery_dict)
    # The big model predicts "hard" (for the small model) examples from all inputs.
    combiner_node = get_combiner_node_eval(more_important_inputs_head, less_important_inputs_head, model_input_node,
                                           threshold_node)
    new_big_model_node, preds_combiner_node = get_big_model_nodes(model_node, combiner_node, big_model, aux_data,
                                                                  threshold_node,
                                                                  small_model_preds_name)
    base_discovery_dict = {}
    # Remove the original code for creating model inputs to replace with the new code.
    training_dependencies = get_model_node_dependencies(model_input_node, base_discovery_dict)
    for node in training_dependencies:
        sorted_nodes.remove(node)
    # Add all the new code for creating model inputs and training from them.
    model_node_index = sorted_nodes.index(model_node)
    sorted_nodes = sorted_nodes[:model_node_index] + more_important_inputs_block + \
                   [new_small_model_node, threshold_node] + less_important_inputs_block + \
                   [combiner_node, new_big_model_node, preds_combiner_node] + sorted_nodes[model_node_index + 1:]
    return sorted_nodes
