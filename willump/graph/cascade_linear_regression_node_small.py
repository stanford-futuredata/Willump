from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.linear_regression_node import LinearRegressionNode

from weld.types import *

from typing import List, Tuple


class CascadeLinearRegressionNodeSmall(LinearRegressionNode):
    """
    Used in cascades.  Predict with linear regression but return a vector of chars where each entry is 1 if
    predicting 1, 0 if predicting 0, and 2 if not confident.  Confidence threshold is an input.
    """

    def __init__(self, input_node: WillumpGraphNode, input_name: str, input_type: WeldType, output_name: str,
                 output_type: WeldType,
                 logit_weights, logit_intercept, aux_data: List[Tuple[int, WeldType]], batch=True) -> None:
        super(CascadeLinearRegressionNodeSmall, self).__init__(input_node, input_name, input_type, output_name, output_type,
                                                               logit_weights, logit_intercept, aux_data, batch)
        self.output_type = WeldVec(WeldChar())
        self.threshold = 1.0

    def get_node_weld(self) -> str:
        if isinstance(self._input_type, WeldCSR):
            weld_program = \
                """
                let row_numbers: vec[i64] = INPUT_NAME.$0;
                let index_numbers: vec[i64] = INPUT_NAME.$1;
                let data = INPUT_NAME.$2;
                let out_len: i64 = INPUT_NAME.$3;
                let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                let base_vector: vec[f64] = result(for(rangeiter(0L, out_len, 1L),
                    appender[f64],
                    | bs, i, x|
                        merge(bs, intercept)
                ));
                let output_probs: vec[f64] = result(for(zip(row_numbers, index_numbers, data),
                    vecmerger[f64,+](base_vector),
                    | bs, i, x |
                        merge(bs, {x.$0, lookup(WEIGHTS_NAME, x.$1) * f64(x.$2)})
                ));
                let OUTPUT_NAME: vec[i8] = result(for(output_probs,
                    appender[i8],
                    | bs, i, x |
                    let prob = 1.0 / (1.0 + exp( -1.0 * f64(x)));
                    # 0 for predict false, 1 for predict true, 2 for not confident
                    let prediction = select(prob > THRESHOLD, 1c, select(prob < 1.0 - THRESHOLD, 0c, 2c));
                    merge(bs, prediction)
                ));
                """
        else:
            assert (isinstance(self._input_type, WeldPandas))
            assert self.batch
            sum_string = ""
            for i in range(len(self._input_type.column_names)):
                sum_string += "lookup(WEIGHTS_NAME, %dL) * f64(lookup(INPUT_NAME.$%d, result_i))+" % (i, i)
            sum_string = sum_string[:-1]
            weld_program = \
                """
                let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                let OUTPUT_NAME: vec[i8] = result(for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                    appender[i8],
                    | results: appender[i8], iter_num: i64, result_i: i64 |
                        let sum: f64 = SUM_STRING + intercept;
                        let prob = 1.0 / (1.0 + exp( -1.0 * f64(sum)));
                        let prediction = select(prob > THRESHOLD, 1c, select(prob < 1.0 - THRESHOLD, 0c, 2c));
                        merge(results, prediction)
                ));
                """
            weld_program = weld_program.replace("SUM_STRING", sum_string)
        weld_program = weld_program.replace("WEIGHTS_NAME", self.weights_data_name)
        weld_program = weld_program.replace("INTERCEPT_NAME", self.intercept_data_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("THRESHOLD", str(self.threshold))
        return weld_program

    def __repr__(self):
        return "Small-model Linear regression node for input {0} output {1}\n" \
            .format(self._input_string_name, self._output_name)
