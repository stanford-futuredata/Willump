from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List, Tuple, Mapping


class PandasSeriesToDataFrameNode(WillumpGraphNode):
    """
    Convert a Series into a (single-row) Dataframe.
    """
    output_type: WeldPandas

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 input_type: WeldSeriesPandas) -> None:
        """
        Initialize the node.
        """
        self._output_name = output_name
        self._input_name = input_name
        self._input_names = [input_name]
        self._input_nodes = [input_node]
        self._input_type = input_type
        assert isinstance(input_type, WeldSeriesPandas)
        self.output_type = WeldPandas(column_names=input_type.column_names,
                                      field_types=[WeldVec(input_type.elemType)] * len(input_type.column_names))

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        conversion_string = ""
        for i in range(len(self._input_type.column_names)):
            conversion_string += "result(merge(appender[ELEM_TYPE](1L), lookup(INPUT_NAME, %dL)))," % i
        weld_program = \
            """
            let OUTPUT_NAME = {CONVERSION_STRING};
            """
        weld_program = weld_program.replace("CONVERSION_STRING", conversion_string)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_name)
        weld_program = weld_program.replace("ELEM_TYPE", str(self._input_type.elemType))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Pandas Series to Dataframe node for input {0} output {1}\n" \
            .format(self._input_name, self._output_name)
