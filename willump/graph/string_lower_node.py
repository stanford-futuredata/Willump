from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class StringLowerNode(WillumpGraphNode):
    """
    Willump String Lowering Node.  Takes in a list of strings (presumably split by a string
    split node) and lowers all capital letters.  Equivalent to mapping Python's lower() over
    a list of strings.
    """
    _input_node: WillumpGraphNode
    _input_string_name: str
    _output_name: str

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str) -> None:
        self._input_node = input_node
        self._input_string_name = input_name
        self._output_name = output_name
        self._input_names = [input_name]

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let OUTPUT_NAME = result(
                for(INPUT_NAME,
                    appender[vec[i8]],
                    | bs: appender[vec[i8]], i: i64, n: vec[i8] |
                        let lowered_n = for(n,
                            appender[i8],
                            | ns: appender[i8], i_c: i64, char: i8 |
                                if(char >= 65c && char <= 90c, # If char is capital
                                    merge(ns, char + 32c), # Lower-case it
                                    merge(ns, char)
                                )  
                        );
                        merge(bs, result(lowered_n))
                )
            );
            """
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "String Lower node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
