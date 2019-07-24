from willump.graph.willump_graph_node import WillumpGraphNode
from typing import List


class CascadeTopKSelectionNode(WillumpGraphNode):
    """
    In the top-K setting, applies a percentile threshold to cascade small model prediction.  Only the top some
    percentile of the input gets sent to the big model.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 top_k: int) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = [input_node]
        self._output_name = output_name
        self._input_names = [input_name]
        self._input_name = input_name
        self._top_k = top_k

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        weld_program = \
            """
            # Run the big model on the top 10 * K samples.
            let percentile: f64 = 1.0 - 10.0 * (f64(TOP_K) / f64(len(INPUT_NAME)));
            # Clamp the percentile of elements above which the big model runs between 0% and 95%.
            let percentile = if(percentile > 0.0, percentile, 0.0);
            let percentile = if(percentile < 0.95, percentile, 0.95);
            let sorted_proba = sort(INPUT_NAME, |x, y| compare(x, y));
            let threshold_index = i64(percentile * f64(len(sorted_proba)));
            let threshold: f64 = lookup(sorted_proba, threshold_index);
            let output_len: i64 = len(INPUT_NAME);
            let OUTPUT_NAME: vec[i8] = result(for(INPUT_NAME,
                appender[i8](output_len),
                | bs, i: i64, x: f64|
                    # 2 if top percentile, 0 otherwise.
                    merge(bs, if(x >= threshold, 2c, 0c))
            ));
            """
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_name)
        weld_program = weld_program.replace("TOP_K", str(self._top_k))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def __repr__(self):
        return "Cascade top-K selection node for input {0} output {1}\n" \
            .format(self._input_name, self._output_name)
