from typing import List

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode


class PandasToDenseMatrixNode(WillumpGraphNode):
    """
    Convert a Series into a (single-row) Dataframe.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 input_type: WeldPandas, output_type: WeldVec) -> None:
        """
        Initialize the node.
        """
        self._output_name = output_name
        self._input_name = input_name
        self._input_names = [input_name]
        self._input_nodes = [input_node]
        self._input_type = input_type
        assert isinstance(input_type, WeldPandas)
        assert isinstance(output_type, WeldVec)
        assert isinstance(output_type.elemType, WeldVec)
        self._elem_type = output_type.elemType.elemType
        self._output_type = output_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        conversion_string = "appender[ELEM_TYPE]"
        for i in range(len(self._input_type.column_names)):
            conversion_string = "merge(%s, ELEM_TYPE(lookup(INPUT_NAME.$%d, x)))" % (conversion_string, i)
        weld_program = \
            """
            let OUTPUT_NAME = result(for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                appender[vec[ELEM_TYPE]],
                |bs, i, x|
                    let new_row = result(CONVERSION_STRING);
                    merge(bs, new_row)
            ));
            """
        weld_program = weld_program.replace("CONVERSION_STRING", conversion_string)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_name)
        weld_program = weld_program.replace("ELEM_TYPE", str(self._elem_type))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def get_output_type(self) -> WeldType:
        return self._output_type

    def get_output_types(self) -> List[WeldType]:
        return [self._output_type]

    def __repr__(self):
        return "Pandas Dataframe to dense matrix node for input {0} output {1}\n" \
            .format(self._input_name, self._output_name)
