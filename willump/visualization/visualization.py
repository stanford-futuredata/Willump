import platform
import subprocess
from typing import List, Set, Tuple, MutableMapping, Mapping

import graphviz

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode


def filter_node(node: WillumpGraphNode) -> bool:
    if "AUX" in node.__repr__():
        return False
    if "re.sub" in node.__repr__():
        return False
    if "return" in node.graphviz_repr():
        return False
    if "Array Concatenation" in node.graphviz_repr():
        return False
    if "return" in node.graphviz_repr():
        return False
    return True


def graph_to_dot_string(graph: WillumpGraph, mi_inputs: List[WillumpGraphNode],
                        li_inputs: List[WillumpGraphNode], indices_to_costs_map, indices_to_importances_map,
                        model_node_input) -> Tuple[str, str]:
    if model_node_input is not None:
        nodes_to_importances: MutableMapping[WillumpGraphNode, float] = {}
        nodes_to_costs: MutableMapping[WillumpGraphNode, int] = {}
        cost_scale_factor = 100 / sum(indices_to_costs_map.values())
        importance_scale_factor = 100 / sum(indices_to_importances_map.values())
        for node, indices in model_node_input.items():
            if not isinstance(indices, tuple):
                indices = tuple(indices.values())
            nodes_to_importances[node] = indices_to_importances_map[indices] * importance_scale_factor
            nodes_to_costs[node] = indices_to_costs_map[indices] * cost_scale_factor
    else:
        nodes_to_importances = {}
        nodes_to_costs = {}

    labels_string: str = ""
    graph_string: str = ""
    if len(mi_inputs) > 0:
        labels_string += "0 [label=\"Approximate Model: Confidence Threshold\"]"
        graph_string += "0 -> {0} [label=\"Above\"];\n".format(abs(graph.get_output_node().__hash__()))
    node_queue: List[WillumpGraphNode] = [graph.get_output_node()]
    nodes_seen: Set[WillumpGraphNode] = set()
    while len(node_queue) > 0:
        current_node = node_queue.pop(0)
        if filter_node(current_node):
            metadata = ""
            color = "black"
            width = 1
            if current_node in mi_inputs:
                color = "darkgreen"
                width = 3
            elif current_node in li_inputs:
                color = "blue"
                width = 3
            if current_node in nodes_to_importances:
                width = 3
                metadata += "\\nCost=%.1f%% Importance=%.1f%%" % (
                nodes_to_costs[current_node], nodes_to_importances[current_node])
            node_label = "{0} [label=\"{1}{4}\", color={2}, penwidth={3}];\n".format(abs(current_node.__hash__()),
                                                                                     current_node.graphviz_repr().replace(
                                                                                         "\n", ""),
                                                                                     color, width, metadata)
            labels_string += node_label
        next_nodes = current_node.get_in_nodes().copy()
        changes = True
        while changes:
            changes = False
            for node in next_nodes:
                if not filter_node(node):
                    for next_node in node.get_in_nodes():
                        next_nodes.append(next_node)
                    next_nodes.remove(node)
                    changes = True
        for node in next_nodes:
            if filter_node(current_node) and filter_node(node):
                if node in mi_inputs:
                    edge_string = "{0} -> 0;\n".format(abs(node.__hash__()))
                    if node is mi_inputs[0]:
                        edge_string += "0 -> {0}[label=\"Below\"];\n".format(abs(current_node.__hash__()))
                else:
                    edge_string = "{1} -> {0};\n".format(abs(current_node.__hash__()), abs(node.__hash__()))
                graph_string += edge_string
            if node not in nodes_seen:
                nodes_seen.add(node)
                node_queue.append(node)
    return labels_string, graph_string


def visualize_graph(graph: WillumpGraph, mi_inputs: List[WillumpGraphNode] = (),
                    li_inputs: List[WillumpGraphNode] = (),
                    indices_to_costs_map: Mapping = None,
                    indices_to_importances_map: Mapping = None, model_node_inputs=None) -> None:
    labels_string, graph_string = graph_to_dot_string(graph, mi_inputs, li_inputs,
                                                      indices_to_costs_map, indices_to_importances_map,
                                                      model_node_inputs)
    dot_string = "digraph G {{\n {0} {1}}}".format(labels_string, graph_string)
    print(dot_string)
    source = graphviz.Source(dot_string)
    if "microsoft" in platform.uname()[3].lower():  # Detect WSL
        source.render(format="png", filename="output")
        subprocess.run(["cmd.exe", "/C", "call", "output.png"])
    else:
        source.render(format="png", filename="output", view=True)
