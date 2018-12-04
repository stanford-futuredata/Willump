from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *
from typing import List


class ArrayAppendNode(WillumpGraphNode):
    """
    Willump Append Node.  Appends a value to the end of an array.
    """
    _input_nodes: List[WillumpGraphNode]
    _input_array_name: str
    _input_value_name: str
    _output_name: str
    _input_array_type: WeldType

    def __init__(self,
                 input_array_node: WillumpGraphNode,
                 input_value_node: WillumpGraphNode,
                 output_name: str,
                 input_array_type: WeldType) -> None:
        self._input_nodes = [input_array_node, input_value_node]
        self._input_array_name = input_array_node.get_output_name()
        self._input_value_name = input_value_node.get_output_name()
        self._output_name = output_name
        self._input_array_type = input_array_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array_append"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let appender_of_in_array = for(INPUT_ARRAY_NAME,
                appender[INPUT_ARRAY_ELEMTYPE],
                | bs: appender[INPUT_ARRAY_ELEMTYPE], i: i64, n: INPUT_ARRAY_ELEMTYPE |
                    merge(bs, n)
            );
            let OUTPUT_NAME = result(merge(appender_of_in_array, INPUT_ARRAY_ELEMTYPE(INPUT_VALUE_NAME)));
            """
        weld_program = weld_program.replace("INPUT_ARRAY_ELEMTYPE", str(self._input_array_type.elemType))
        weld_program = weld_program.replace("INPUT_ARRAY_NAME", self._input_array_name)
        weld_program = weld_program.replace("INPUT_VALUE_NAME", self._input_value_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "String Lower node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
