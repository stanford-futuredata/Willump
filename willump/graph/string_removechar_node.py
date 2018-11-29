from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class StringRemoveCharNode(WillumpGraphNode):
    """
    Willump Char Removing Node.  Takes in a list of strings (presumably split by a string
    split node) and removes all instance of some character from each word in the list.
    """
    _input_node: WillumpGraphNode
    _input_string_name: str
    _output_name: str
    _target_char: int

    def __init__(self, input_node: WillumpGraphNode, target_char: str, output_name: str) -> None:
        self._input_node = input_node
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name
        assert(len(target_char) == 1)
        self._target_char = ord(target_char)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_node_type(self) -> str:
        return "string_remove_char"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let OUTPUT_NAME = result(
                for(INPUT_NAME,
                    appender[vec[i8]],
                    | bs: appender[vec[i8]], i: i64, n: vec[i8] |
                        let remove_char_n = for(n,
                            appender[i8],
                            | ns: appender[i8], i_c: i64, char: i8 |
                                if(char == TARGET_CHARc,
                                    ns, # Remove it
                                    merge(ns, char)
                                )  
                        );
                        merge(bs, result(remove_char_n))
                )
            );
            """
        weld_program = weld_program.replace("TARGET_CHAR", str(self._target_char))
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Char Remove node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
