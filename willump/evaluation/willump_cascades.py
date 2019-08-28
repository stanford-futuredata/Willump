from willump import *
from willump.evaluation.willump_cascades_utilities import *
from willump.graph.cascade_column_selection_node import CascadeColumnSelectionNode
from willump.graph.cascade_combine_predictions_node import CascadeCombinePredictionsNode
from willump.graph.cascade_point_early_exit_node import CascadePointEarlyExitNode
from willump.graph.cascade_stack_dense_node import CascadeStackDenseNode
from willump.graph.cascade_stack_sparse_node import CascadeStackSparseNode
from willump.graph.cascade_threshold_proba_node import CascadeThresholdProbaNode
from willump.graph.cascade_topk_selection_node import CascadeTopKSelectionNode
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_model_node import WillumpModelNode
from willump.graph.willump_predict_node import WillumpPredictNode
from willump.graph.willump_predict_proba_node import WillumpPredictProbaNode
from willump.graph.willump_training_node import WillumpTrainingNode
from willump.willump_utilities import *

import scipy.sparse


def training_model_cascade_pass(graph: WillumpGraph,
                                training_cascades: dict,
                                train_predict_score_functions: tuple,
                                top_k: int) -> None:
    """
    Take in a program training a model.  Rank features in the model by importance.  Partition
    features into "more important" and "less important."  Train a small model on only more important features
    and a big model on all features.
    """
    sorted_nodes = wg_passes.topological_sort_graph(graph)
    sorted_nodes = wg_passes.push_back_python_nodes_pass(sorted_nodes)
    wg_passes.model_input_identification_pass(sorted_nodes)
    for node in sorted_nodes:
        if isinstance(node, WillumpTrainingNode):
            training_node: WillumpTrainingNode = node
            break
    else:
        panic("No Model in Training Function")
        return None
    train_x, train_y = training_node.get_train_x_y()
    if isinstance(train_x, pd.DataFrame):
        train_x = train_x.values
    if isinstance(train_y, pd.DataFrame) or isinstance(train_y, pd.Series):
        train_y = train_y.values
    feature_importances, orig_model = calculate_feature_importance(x=train_x, y=train_y,
                                                                   train_predict_score_functions=train_predict_score_functions,
                                                                   model_inputs=training_node.get_model_inputs())
    training_cascades["feature_importances"] = feature_importances
    indices_to_costs_map = create_indices_to_costs_map(training_node)
    training_cascades["indices_to_costs_map"] = indices_to_costs_map
    min_cost = np.inf
    more_important_inputs, less_important_inputs = None, None
    last_mi_candidate_length = 0
    for cost_cutoff in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        _, _, mi_indices, li_indices, mi_cost, t_cost = split_model_inputs(training_node,
                                                               feature_importances,
                                                               indices_to_costs_map,
                                                               cost_cutoff)
        if len(mi_indices) == last_mi_candidate_length:
            continue
        last_mi_candidate_length = len(mi_indices)
        if mi_cost == 0.0:
            continue
        if top_k is not None:
            valid_size = train_x.shape[0] // 4
            threshold, cost = \
                calculate_feature_set_performance_top_k(x=train_x, y=train_y,
                                                        train_predict_score_functions=train_predict_score_functions,
                                                        mi_feature_indices=mi_indices, mi_cost=mi_cost,
                                                        total_cost=t_cost, top_k_distribution=[top_k],
                                                        valid_size_distribution=[valid_size])
        else:
            threshold, cost = \
                calculate_feature_set_performance(x=train_x, y=train_y,
                                                  train_predict_score_functions=train_predict_score_functions,
                                                  mi_feature_indices=mi_indices, orig_model=orig_model, mi_cost=mi_cost,
                                                  total_cost=t_cost)
        print(cost_cutoff, threshold, cost)
        if cost < min_cost:
            more_important_inputs = sorted(mi_indices)
            less_important_inputs = sorted(li_indices)
            training_cascades["cascade_threshold"] = threshold
            training_cascades["cost_cutoff"] = cost_cutoff
            min_cost = cost
    willump_train_function, _, _, _ = train_predict_score_functions
    mi_train_x = train_x[:, more_important_inputs]
    li_train_x = train_x[:, less_important_inputs]
    training_cascades["small_model"] = willump_train_function(mi_train_x, train_y)
    if scipy.sparse.issparse(train_x):
        full_model_train_x = scipy.sparse.hstack((mi_train_x, li_train_x))
    else:
        full_model_train_x = np.concatenate((mi_train_x, li_train_x), axis=1)
    training_cascades["big_model"] = willump_train_function(full_model_train_x, train_y)
    return None


def eval_model_cascade_pass(sorted_nodes: List[WillumpGraphNode],
                            typing_map: MutableMapping[str, WeldType],
                            eval_cascades: dict,
                            cascade_threshold: float,
                            batch: bool,
                            top_k: Optional[int]) -> List[WillumpGraphNode]:
    """
    Take in a program with a model.  Use pre-computed feature importances to partition the models' input
    sources into those more and less important.  Rewrite the program so it first evaluates a pre-trained smaller
    model on all more-important input sources and checks confidence.  If it is above some threshold, use
    that prediction, otherwise, predict with a pre-trained bigger model on all inputs.
    """

    def get_small_model_nodes(orig_model_node: WillumpModelNode, new_input_node: WillumpGraphNode) \
            -> Tuple[WillumpModelNode, WillumpGraphNode]:
        assert (isinstance(orig_model_node, WillumpPredictNode) or
                (top_k is not None and isinstance(orig_model_node, WillumpPredictProbaNode)))
        assert (len(new_input_node.get_output_names()) == 1)
        proba_output_name = "small__proba_" + orig_model_node.get_output_name()
        output_type = WeldVec(WeldDouble())
        typing_map[proba_output_name] = output_type
        new_input_name = new_input_node.get_output_names()[0]
        predict_proba_node = WillumpPredictProbaNode(model_name=SMALL_MODEL_NAME,
                                                     x_name=new_input_name,
                                                     x_node=new_input_node,
                                                     output_name=proba_output_name,
                                                     output_type=output_type,
                                                     input_width=orig_model_node.input_width)
        threshold_output_name = "small_preds_" + orig_model_node.get_output_name()
        if top_k is None:
            threshold_node = CascadeThresholdProbaNode(input_node=predict_proba_node, input_name=proba_output_name,
                                                       output_name=threshold_output_name,
                                                       threshold=cascade_threshold)
        else:
            threshold_node = CascadeTopKSelectionNode(input_node=predict_proba_node, input_name=proba_output_name,
                                                      output_name=threshold_output_name,
                                                      top_k=top_k * cascade_threshold)
        typing_map[threshold_output_name] = WeldVec(WeldChar())
        return predict_proba_node, threshold_node

    def get_big_model_nodes(orig_model_node: WillumpModelNode, new_input_node: WillumpGraphNode,
                            small_model_output_node: CascadeThresholdProbaNode, small_model_output_name: str) \
            -> Tuple[WillumpModelNode, CascadeCombinePredictionsNode]:
        assert (isinstance(orig_model_node, WillumpPredictNode) or
                (top_k is not None and isinstance(orig_model_node, WillumpPredictProbaNode)))
        assert (len(new_input_node.get_output_names()) == 1)
        output_name = orig_model_node.get_output_name()
        output_type = orig_model_node.get_output_type()
        new_input_name = new_input_node.get_output_names()[0]
        if top_k is None:
            big_model_output = WillumpPredictNode(model_name=BIG_MODEL_NAME,
                                                  x_name=new_input_name,
                                                  x_node=new_input_node,
                                                  output_name=output_name,
                                                  output_type=output_type,
                                                  input_width=orig_model_node.input_width)
        else:
            big_model_output = WillumpPredictProbaNode(model_name=BIG_MODEL_NAME,
                                                       x_name=new_input_name,
                                                       x_node=new_input_node,
                                                       output_name=output_name,
                                                       output_type=output_type,
                                                       input_width=orig_model_node.input_width)
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
            mi_output_types = mi_head.get_output_types()[0]
            li_output_types = li_head.get_output_types()[0]
            assert (isinstance(mi_output_types, WeldPandas) and isinstance(li_output_types, WeldPandas))
            new_selected_columns = mi_output_types.column_names + li_output_types.column_names
            new_field_types = mi_output_types.field_types + li_output_types.field_types
            output_name = orig_node.get_output_names()[0]
            new_output_type = WeldPandas(column_names=new_selected_columns, field_types=new_field_types)
            typing_map[output_name] = new_output_type
            return CascadeColumnSelectionNode(more_important_nodes=[mi_head],
                                              more_important_names=[mi_head.get_output_names()[0]],
                                              more_important_types=[mi_output_types],
                                              less_important_nodes=[li_head],
                                              less_important_names=[li_head.get_output_names()[0]],
                                              less_important_types=[li_output_types],
                                              output_name=output_name,
                                              small_model_output_node=small_model_output_node,
                                              small_model_output_name=small_model_output_node.get_output_name(),
                                              selected_columns=new_selected_columns)
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
    indices_to_costs_map = eval_cascades["indices_to_costs_map"]
    if cascade_threshold is None:
        cascade_threshold = eval_cascades["cascade_threshold"]
    cost_cutoff = eval_cascades["cost_cutoff"]
    more_important_inputs, less_important_inputs, _, _, _, _ = split_model_inputs(model_node, feature_importances,
                                                                               indices_to_costs_map, cost_cutoff)
    model_input_node = model_node.get_in_nodes()[0]
    # Create Willump graphs and code blocks that produce the more important inputs.
    base_discovery_dict = {}
    more_important_inputs_head = graph_from_input_sources(model_input_node, more_important_inputs, typing_map,
                                                          base_discovery_dict, "more")
    more_important_inputs_block = get_model_node_dependencies(more_important_inputs_head, base_discovery_dict)
    # The small model predicts all examples from the more important inputs.
    new_small_model_node, threshold_node = \
        get_small_model_nodes(model_node, more_important_inputs_head)
    small_model_preds_name = threshold_node.get_output_name()
    # Less important inputs are materialized only if the small model lacks confidence in an example.
    base_discovery_dict = {}
    less_important_inputs_head = graph_from_input_sources(model_input_node, less_important_inputs, typing_map,
                                                          base_discovery_dict, "less",
                                                          small_model_output_node=threshold_node)
    less_important_inputs_block = get_model_node_dependencies(less_important_inputs_head, base_discovery_dict,
                                                              small_model_output_node=threshold_node)
    # The big model predicts "hard" (for the small model) examples from all inputs.
    combiner_node = get_combiner_node_eval(more_important_inputs_head, less_important_inputs_head, model_input_node,
                                           threshold_node)
    new_big_model_node, preds_combiner_node = get_big_model_nodes(model_node, combiner_node,
                                                                  threshold_node,
                                                                  small_model_preds_name)
    base_discovery_dict = {}
    # Remove the original code for creating model inputs to replace with the new code.
    training_dependencies = get_model_node_dependencies(model_input_node, base_discovery_dict)
    for node in training_dependencies:
        sorted_nodes.remove(node)
    # Add all the new code for creating model inputs and training from them.
    model_node_index = sorted_nodes.index(model_node)
    small_model_nodes = [new_small_model_node, threshold_node]
    # In a point setting, you can immediately exit if the small model node is confident.
    if batch is False:
        point_early_exit_node = CascadePointEarlyExitNode(small_model_output_node=threshold_node,
                                                          small_model_output_name=small_model_preds_name)
        small_model_nodes.append(point_early_exit_node)
    big_model_nodes = [combiner_node, new_big_model_node, preds_combiner_node]
    sorted_nodes = sorted_nodes[:model_node_index] + more_important_inputs_block + small_model_nodes + \
                   less_important_inputs_block + big_model_nodes + sorted_nodes[model_node_index + 1:]
    return sorted_nodes
