import typing
import willump.graph.willump_graph_node as wgn


class TransformAdditionNode(wgn.WillumpGraphNode):
    """
    Willump addition transform node.  Takes in a vector of numbers and two input indexes, returns
    a vector consisting of the sum of the inputs.
    """
    input_nodes: typing.List[wgn.WillumpGraphNode]
    add_input_one_index: int
    add_input_two_index: int

    def __init__(self, input_node: wgn.WillumpGraphNode, add_input_one_index: int,
                 add_input_two_index: int) -> None:
        self.input_nodes = [input_node]
        self.add_input_one_index = add_input_one_index
        self.add_input_two_index = add_input_two_index

    def get_in_nodes(self) -> typing.List[wgn.WillumpGraphNode]:
        return self.input_nodes

    def get_node_type(self) -> str:
        return "addition"

    def get_node_weld(self) -> str:
        weld_str: str = \
            "let b = appender[f64];" \
            "let temp = lookup({0}, {1}L) + lookup({0}, {2}L);" \
            "let b2 = merge(b, temp);" \
            "result(b2)"
        return weld_str.format("{0}", self.add_input_one_index, self.add_input_two_index)