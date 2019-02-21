from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List


class PandasSeriesConcatenationNode(WillumpGraphNode):
    """
    Willump Pandas Series Concatenation Node.  Horizontally concatenates Pandas series.
    """

    input_types: List[WeldSeriesPandas]
    output_type: WeldSeriesPandas

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], input_types: List[WeldSeriesPandas],
                 output_name: str, output_type: WeldSeriesPandas) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = input_nodes
        self._output_name = output_name
        assert(isinstance(output_type, WeldSeriesPandas))
        self.output_type = output_type
        self.input_types = input_types
        self._input_names = input_names

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let pre_output = appender[ELEM_TYPE];
            """
        for input_name in self._input_names:
            weld_program += \
                """
                let pre_output = for(%s,
                    pre_output,
                    | bs, i, x|
                    merge(pre_output, ELEM_TYPE(x))
                );
                """ % input_name
        weld_program += \
            """
            let OUTPUT_NAME = result(pre_output);
            """
        weld_program = weld_program.replace("ELEM_TYPE", str(self.output_type.elemType))
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Pandas Series concatenation node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
