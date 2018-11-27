from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class StringSplitNode(WillumpGraphNode):
    """
    Willump String Splitting node.  Implements the semantics of Python's string.split function
    without arguments, splitting the input string on all whitespace.  Used as a tokenizer for now.

    TODO:  Support delimiter and max-split arguments.
    """
    _input_node: WillumpGraphNode
    _input_string_name: str
    _output_name: str

    def __init__(self, input_node: WillumpGraphNode, output_name: str) -> None:
        self._input_node = input_node
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_node_type(self) -> str:
        return "string_split"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let struct_all = for(
                INPUT_NAME,
                {appender[vec[i8]], appender[i8], merge(merger[i8, +], 32c)},
                | bs: {appender[vec[i8]], appender[i8], merger[i8, +]}, i: i64, n: i8|
                    if ( n == 9c || n == 10c || n == 11c || n == 12c || n == 13c || n == 32c,
                        let nprev = result(bs.$2);
                        if ( nprev == 9c || nprev == 10c || nprev == 11c || nprev == 12c || nprev == 13c || nprev == 32c,
                            {bs.$0, appender[i8], merge(merger[i8, +], n)},
                            {merge(bs.$0, result(merge(bs.$1, 0c))), appender[i8], merge(merger[i8, +], n)}),
                        {bs.$0, merge(bs.$1, n), merge(merger[i8, +], n)}
                    )
            );
            let nlast = result(struct_all.$2);
            let final_struct = if (nlast == 9c || nlast == 10c || nlast == 11c || nlast == 12c || nlast == 13c || nlast == 32c,
                                    struct_all.$0,
                                    merge(struct_all.$0, result(merge(struct_all.$1, 0c))));
            let OUTPUT_NAME = result(final_struct);
            """
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Input node for input {0}\n".format(self._arg_name)
