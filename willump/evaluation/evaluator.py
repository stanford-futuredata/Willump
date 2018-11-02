import willump.graph.willump_graph as wg
import willump.graph.willump_graph_node as wgn

import weld.weldobject
from weld.types import *
import weld.encoders

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()


def graph_to_weld(graph: wg.WillumpGraph) -> str:
    """
    Convert a Willump graph into a valid Weld program string.

    For now assume graph is a linked list where each node has one input.
    """
    current_node: wgn.WillumpGraphNode = graph.get_output_node()
    weld_str: str = "{0}"
    while len(current_node.get_in_nodes()) > 0:
        assert(len(current_node.get_in_nodes()) == 1)
        node_type: str = current_node.get_node_type()
        if node_type == "math":
            weld_str = weld_str.format(current_node.get_node_weld())
        else:
            pass
        current_node = current_node.get_in_nodes()[0]
    return weld_str


def evaluate_weld(weld_program: str, input_vector):
    """
    Evaluate a Weld program string on a Numpy input vector.  TODO:  Support multiple input vectors.
    """
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    vec_name = weld_object.update(input_vector, tys=WeldVec(WeldDouble()))
    weld_object.weld_code = weld_program.format(vec_name)
    return weld_object.evaluate(WeldVec(WeldDouble()), verbose=False)



