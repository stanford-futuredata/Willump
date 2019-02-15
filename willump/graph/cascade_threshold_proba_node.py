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
            let df_size = len(OUTPUT_NAME);
            let pre_mapping = iterate({0L, 0L, dictmerger[i64, i64, +]},
                | input |
                    let more_important_iter = input.$0;
                    let less_important_iter = input.$1;
                    let output_dict = input.$2;
                    if(more_important_iter == df_size,
                        {{more_important_iter, less_important_iter, output_dict}, false},
                        let small_model_output = lookup(OUTPUT_NAME, more_important_iter);
                        if(small_model_output != 2c,
                            {{more_important_iter + 1L, less_important_iter, output_dict}, true},
                            {{more_important_iter + 1L, less_important_iter + 1L, merge(output_dict, {more_important_iter, less_important_iter})}, true}
                        )
                    )
            );  
            let OUTPUT_NAME_len = pre_mapping.$1;
            let OUTPUT_NAME_mapping = result(pre_mapping.$2);
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
