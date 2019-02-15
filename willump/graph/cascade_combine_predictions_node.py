from willump.graph.willump_graph_node import WillumpGraphNode
from typing import List
from weld.types import *


class CascadeCombinePredictionsNode(WillumpGraphNode):
    """
    Applies a threshold to cascade small model confidence predictions.  Returns a vector of i8 where each entry
    is 0c if the small model predicted false, 1c if true, or 2c if not confident.
    """

    def __init__(self, big_model_predictions_node: WillumpGraphNode, big_model_predictions_name: str,
                 small_model_predictions_node: WillumpGraphNode, small_model_predictions_name: str,
                 output_name: str, output_type: WeldType) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = [big_model_predictions_node, small_model_predictions_node]
        self._output_name = output_name
        self._input_names = [big_model_predictions_name, small_model_predictions_name]
        self._big_model_predictions_name = big_model_predictions_name
        self._small_model_predictions_name = small_model_predictions_name
        self._output_type = output_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        assert(isinstance(self._output_type, WeldVec))
        elem_type = self._output_type.elemType
        weld_program = \
            """
            let output_len: i64 = len(SMALL_MODEL_OUTPUT);
            let pre_output = iterate({0L, 0L, appender[ELEM_TYPE](output_len)},
                | input |
                    let more_important_iter = input.$0;
                    let less_important_iter = input.$1;
                    let output_array = input.$2;
                    if(more_important_iter == output_len,
                        {{more_important_iter, less_important_iter, output_array}, false},
                        let small_model_output = lookup(SMALL_MODEL_OUTPUT, more_important_iter);
                        if(small_model_output != 2c,
                            {{more_important_iter + 1L, less_important_iter, merge(output_array, ELEM_TYPE(small_model_output))}, true},
                            {{more_important_iter + 1L, less_important_iter + 1L, merge(output_array, ELEM_TYPE(lookup(BIG_MODEL_OUTPUT, less_important_iter)))}, true}
                        )
                    )
            );  
            let OUTPUT_NAME: vec[ELEM_TYPE] = result(pre_output.$2);
            """
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("SMALL_MODEL_OUTPUT", self._small_model_predictions_name)
        weld_program = weld_program.replace("BIG_MODEL_OUTPUT", self._big_model_predictions_name)
        weld_program = weld_program.replace("ELEM_TYPE", str(elem_type))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Cascade prediction-combining node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
