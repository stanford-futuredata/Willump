from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List


class StackSparseNode(WillumpGraphNode):
    """
    Willump Stack Sparse node.  Horizontally stacks multiple sparse matrices.
    """

    def __init__(self, input_nodes: List[WillumpGraphNode], output_name: str, output_type: WeldType) -> None:
        """
        Initialize the node.
        """
        self.input_nodes = input_nodes
        self.output_name = output_name
        assert(isinstance(output_type, WeldCSR))
        self._elem_type = output_type.elemType

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self.input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let stacker = {appender[i64], appender[i64], appender[ELEM_TYPE]};
            let stack_width: i64 = 0L;
            """
        for input_node in self.input_nodes:
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
            weld_program = weld_program.replace("INPUT_NAME", input_node.get_output_name())
        weld_program += \
            """
            let OUTPUT_NAME: {vec[i64], vec[i64], vec[ELEM_TYPE], i64, i64} = {result(stacker.$0), 
                result(stacker.$1), result(stacker.$2), stack_height, stack_width};
            """
        weld_program = weld_program.replace("OUTPUT_NAME", self.output_name)
        weld_program = weld_program.replace("ELEM_TYPE", str(self._elem_type))
        return weld_program

    def get_output_name(self) -> str:
        return self.output_name

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n" \
            .format(self.input_nodes, self.output_name)
