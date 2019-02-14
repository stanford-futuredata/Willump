from willump.graph.willump_graph_node import WillumpGraphNode
from typing import List


class CascadeThresholdProbaNode(WillumpGraphNode):
    """
    Applies a threshold to cascade small model confidence predictions.  Returns a vector of i8 where each entry
    is 0c if the small model predicted false, 1c if true, or 2c if not confident.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 threshold: float) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = [input_node]
        self._output_name = output_name
        self._input_names = [input_name]
        self._input_name = input_name
        self._threshold = threshold

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let output_len: i64 = len(INPUT_NAME);
            let OUTPUT_NAME: vec[i8] = result(for(INPUT_NAME,
                appender[i8](output_len),
                | bs, i: i64, x: f64|
                    # 0 for predict false, 1 for predict true, 2 for not confident
                    merge(bs, select(x > THRESHOLD, 1c, select(x < 1.0 - THRESHOLD, 0c, 2c)))
            ));
            """
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_name)
        weld_program = weld_program.replace("THRESHOLD", str(self._threshold))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Cascade threshold node for input {0} output {1}\n" \
            .format(self._input_name, self._output_name)
