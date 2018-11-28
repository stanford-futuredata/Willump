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
            let input_len = len(INPUT_NAME);
            let struct_combined = iterate(
                {appender[vec[i8]], 0L, 0L}, # Appender, index, length of current word
                | iteration: {appender[vec[i8]], i64, i64} |
                    if ( iteration.$1 == input_len,
                        if (iteration.$2 > 0L,
                            { {merge(iteration.$0, slice(INPUT_NAME, iteration.$1 - iteration.$2, 
                                iteration.$2)), iteration.$1 + 1L, 0L}, false},
                            { iteration, false }
                        ),
                        let curr_char = lookup(INPUT_NAME, iteration.$1);
                        if ( (curr_char >= 9c && curr_char <= 13c) || curr_char == 32c, # Whitespace
                            if (iteration.$2 > 0L,
                                { {merge(iteration.$0, slice(INPUT_NAME, iteration.$1 - 
                                    iteration.$2, iteration.$2)), iteration.$1 + 1L, 0L}, true},
                                { {iteration.$0, iteration.$1 + 1L, 0L}, true}
                            ),
                            { {iteration.$0, iteration.$1 + 1L, iteration.$2 + 1L}, true}
                        )
                    )
            );
            let OUTPUT_NAME = result(struct_combined.$0);
            """
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Input node for input {0}\n".format(self._arg_name)
