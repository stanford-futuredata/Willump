from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_python_node import WillumpPythonNode
from typing import List, Set, Tuple


def filter_node(node: WillumpGraphNode) -> bool:
    if "AUX" in node.__repr__():
        return False
    if "re.sub" in node.__repr__():
        return False
    return True


def graph_to_dot_string(graph: WillumpGraph) -> Tuple[str, str]:
    labels_string: str = ""
    graph_string: str = ""
    node_queue: List[WillumpGraphNode] = [graph.get_output_node()]
    nodes_seen: Set[WillumpGraphNode] = set()
    while len(node_queue) > 0:
        current_node = node_queue.pop(0)
        if filter_node(current_node):
            if isinstance(current_node, WillumpPythonNode):
                color = "darkgreen"
            else:
                color = "black"
            node_label = "{0} [label=\"{1}\", color={2}];\n".format(abs(current_node.__hash__()),
                                                                    current_node.graphviz_repr().replace("\n", ""),
                                                                    color)
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
                edge_string = "{1} -> {0};\n".format(abs(current_node.__hash__()), abs(node.__hash__()))
                graph_string += edge_string
            if node not in nodes_seen:
                nodes_seen.add(node)
                node_queue.append(node)
    return labels_string, graph_string


def visualize_graph(graph: WillumpGraph) -> None:
    labels_string, graph_string = graph_to_dot_string(graph)
    dot_string = "digraph G {{\n {0} {1}}}".format(labels_string, graph_string)
    print(dot_string)
