from typing import List
from typing import Tuple, Optional

from weld.types import *

from willump.graph.willump_graph_node import WillumpGraphNode


class StackSparseNode(WillumpGraphNode):
    """
    Willump Stack Sparse node.  Horizontally stacks multiple sparse matrices.
    """

    # Protected Model-Pushing Variables
    _model_type: Optional[str] = None
    _model_parameters: Tuple
    _start_index: int

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], output_name: str,
                 output_type: WeldType) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = input_nodes
        self._output_name = output_name
        assert(isinstance(output_type, WeldCSR))
        self._elem_type = output_type.elemType
        self._output_type = output_type
        self._input_names = input_names

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        num_rows = ""
        for input_name in self._input_names:
            num_rows += "len(%s.$0)+" % input_name
        num_rows = num_rows[:-1]
        weld_program = \
            """
            let num_rows: i64 = NUM_ROWS;
            let stacker = {appender[i64](num_rows), appender[i64](num_rows), appender[ELEM_TYPE](num_rows)};
            let stack_width: i64 = 0L;
            """
        for input_node, input_name in zip(self._input_nodes, self._input_names):
            weld_program += \
                """
                let stacker = for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                    stacker,
                    | bs, i, x: i64 |
                    {merge(bs.$0, lookup(INPUT_NAME.$0, x)), 
                      merge(bs.$1, lookup(INPUT_NAME.$1, x) + stack_width), 
                      merge(bs.$2, ELEM_TYPE(lookup(INPUT_NAME.$2, x)))
                    }
                );
                let stack_height = INPUT_NAME.$3;
                let stack_width = stack_width + INPUT_NAME.$4;
                """
            weld_program = weld_program.replace("INPUT_NAME", input_name)
        weld_program += \
            """
            let OUTPUT_NAME: {vec[i64], vec[i64], vec[ELEM_TYPE], i64, i64} = {result(stacker.$0), 
                result(stacker.$1), result(stacker.$2), stack_height, stack_width};
            """
        weld_program = weld_program.replace("NUM_ROWS", num_rows)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
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
        return "Stack sparse node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
