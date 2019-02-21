from typing import List

from willump.graph.willump_graph_node import WillumpGraphNode


class IdentityNode(WillumpGraphNode):
    """
    Identity node.  Sets one variable equal to another.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str) -> None:
        self._input_node = input_node
        self._output_name = output_name
        self._input_name = input_name

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_in_names(self) -> List[str]:
        return [self._input_name]

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let %s = %s;
            """ % (self._output_name, self._input_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Identity node for input {0} output {1}\n"\
            .format(self._input_name, self._output_name)
