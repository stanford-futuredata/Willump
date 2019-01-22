from willump.graph.willump_graph_node import WillumpGraphNode

from typing import List


class ArraySpaceCombinerNode(WillumpGraphNode):
    """
    Willump Array Space Combiner Node.  Combines all sequences of whitespace in a string into a single space.
    """

    def __init__(self, input_node: WillumpGraphNode, output_name: str) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = [input_node]
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "combine white space"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let OUTPUT_NAME = result(for(INPUT_NAME,
                appender[vec[i8]],
                | results: appender[vec[i8]], i: i64, sentence: vec[i8] |
                    let p_s: vec[i8] = result(for(sentence,
                        appender[i8],
                        |bs, i_s, ch|
                            if((ch >= 9c && ch <= 13c) || ch == 32c, # Whitespace
                                if(i_s > 0L,
                                    let prev_ch = lookup(sentence, i_s - 1L);
                                    if((prev_ch >= 9c && prev_ch <= 13c) || prev_ch == 32c, # Whitespace
                                        bs,
                                        merge(bs, 32c)
                                    ),
                                    merge(bs, 32c)
                                ),
                                merge(bs, ch)
                            )
                    ));
                    merge(results, p_s)
            ));
            """
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Array space combiner node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
