from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode

from weld.types import *

from typing import List, Tuple, Mapping
import importlib
import numpy


class LinearRegressionNode(WillumpGraphNode):
    """
    Willump Linear Regression node.  Evaluates a logistic regression model on an input.

    TODO:  Allow multi-class classification, not just binary.
    """
    weights_data_name: str
    intercept_data_name: str
    input_width: int
    batch: bool

    def __init__(self, input_node: WillumpGraphNode, input_name: str, input_type: WeldType, output_name: str,
                 output_type: WeldType,
                 logit_weights, logit_intercept, aux_data: List[Tuple[int, WeldType]], batch=True) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_string_name = input_name
        self._output_name = output_name
        self.weights_data_name = "AUX_DATA_{0}".format(len(aux_data))
        self.intercept_data_name = "AUX_DATA_{0}".format(len(aux_data) + 1)
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self.weights_data_name))
        self._input_nodes.append(WillumpInputNode(self.intercept_data_name))
        self._input_names = [input_name, self.weights_data_name, self.intercept_data_name]
        self._input_type = input_type
        self._output_type = output_type
        self.batch = batch
        self.input_width = len(logit_weights[0])
        for entry in self._process_aux_data(logit_weights, logit_intercept):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    @staticmethod
    def _process_aux_data(logit_weights, logit_intercept) -> List[Tuple[int, WeldType]]:
        """
        Returns pointers to Weld vectors containing the logistic regression model weights
        and intercept.
        """
        weld_program = \
            """
            true
            """
        import willump.evaluation.willump_executor as wexec
        module_name = wexec.compile_weld_program(weld_program, {},
                                                 input_names=[],
                                                 output_names=[],
                                                 base_filename="encode_logistic_regression_model")
        encode_logit_model = importlib.import_module(module_name)
        weld_weights, weld_intercept = \
            encode_logit_model.caller_func(logit_weights[0], logit_intercept)
        return [(weld_weights, WeldVec(WeldDouble())), (weld_intercept, WeldVec(WeldDouble()))]

    def get_node_weld(self) -> str:
        assert (isinstance(self._output_type, WeldVec))
        output_elem_type_str = str(self._output_type.elemType)
        if isinstance(self._input_type, WeldVec):
            elem_type = self._input_type.elemType.elemType
            weld_program = \
                """
                let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                let OUTPUT_NAME: vec[OUTPUT_TYPE] = result(for(INPUT_NAME,
                    appender[OUTPUT_TYPE],
                    | results: appender[OUTPUT_TYPE], result_i: i64, row: vec[ELEM_TYPE] |
                        let sum: f64 = result(for(zip(row, WEIGHTS_NAME),
                            merger[f64, +],
                            | bs: merger[f64, +], i: i64, inp_weight: {ELEM_TYPE, f64} |
                                merge(bs, f64(inp_weight.$0) * inp_weight.$1)
                        ));
                        merge(results, select(sum + intercept > 0.0, OUTPUT_TYPE(1), OUTPUT_TYPE(0)))
                ));
                """
            weld_program = weld_program.replace("OUTPUT_TYPE", output_elem_type_str)
            weld_program = weld_program.replace("WEIGHTS_NAME", self.weights_data_name)
            weld_program = weld_program.replace("INTERCEPT_NAME", self.intercept_data_name)
            weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
            weld_program = weld_program.replace("ELEM_TYPE", str(elem_type))
        elif isinstance(self._input_type, WeldCSR):
            elem_type = self._input_type.elemType
            weld_program = \
                """
                let row_numbers: vec[i64] = INPUT_NAME.$0;
                let index_numbers: vec[i64] = INPUT_NAME.$1;
                let data: vec[ELEM_TYPE] = INPUT_NAME.$2;
                let out_len: i64 = INPUT_NAME.$3;
                let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                let base_vector: vec[f64] = result(for(rangeiter(0L, out_len, 1L),
                    appender[f64],
                    | bs, i, x|
                        merge(bs, intercept)
                ));
                let output_probs: vec[f64] = result(for(zip(row_numbers, index_numbers, data),
                    vecmerger[f64,+](base_vector),
                    | bs, i, x: {i64, i64, ELEM_TYPE} |
                        merge(bs, {x.$0, lookup(WEIGHTS_NAME, x.$1) * f64(x.$2)})
                ));
                let OUTPUT_NAME: vec[OUTPUT_TYPE] = result(for(output_probs,
                    appender[OUTPUT_TYPE],
                    | bs, i, x |
                    merge(bs, select(x > 0.0, OUTPUT_TYPE(1), OUTPUT_TYPE(0)))
                ));
                """
            weld_program = weld_program.replace("OUTPUT_TYPE", output_elem_type_str)
            weld_program = weld_program.replace("WEIGHTS_NAME", self.weights_data_name)
            weld_program = weld_program.replace("INTERCEPT_NAME", self.intercept_data_name)
            weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
            weld_program = weld_program.replace("ELEM_TYPE", str(elem_type))
        else:
            assert (isinstance(self._input_type, WeldPandas))
            if self.batch:
                sum_string = ""
                for i in range(len(self._input_type.column_names)):
                    sum_string += "lookup(WEIGHTS_NAME, %dL) * f64(lookup(INPUT_NAME.$%d, result_i))+" % (i, i)
                sum_string = sum_string[:-1]
                weld_program = \
                    """
                    let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                    let OUTPUT_NAME: vec[OUTPUT_TYPE] = result(for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                        appender[OUTPUT_TYPE],
                        | results: appender[OUTPUT_TYPE], iter_num: i64, result_i: i64 |
                            let sum: f64 = SUM_STRING;
                            merge(results, select(sum + intercept > 0.0, OUTPUT_TYPE(1), OUTPUT_TYPE(0)))
                    ));
                    """
            else:
                sum_string = ""
                for i in range(len(self._input_type.column_names)):
                    sum_string += "lookup(WEIGHTS_NAME, %dL) * f64(INPUT_NAME.$%d)+" % (i, i)
                sum_string = sum_string[:-1]
                weld_program = \
                    """
                    let intercept: f64 = lookup(INTERCEPT_NAME, 0L);
                    let sum: f64 = SUM_STRING;
                    let OUTPUT_NAME: vec[OUTPUT_TYPE] = result(merge(appender[OUTPUT_TYPE], 
                        select(sum + intercept > 0.0, OUTPUT_TYPE(1), OUTPUT_TYPE(0))));
                    """
            weld_program = weld_program.replace("SUM_STRING", sum_string)
            weld_program = weld_program.replace("OUTPUT_TYPE", output_elem_type_str)
            weld_program = weld_program.replace("WEIGHTS_NAME", self.weights_data_name)
            weld_program = weld_program.replace("INTERCEPT_NAME", self.intercept_data_name)
            weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
            weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Linear regression node for input {0} output {1}\n" \
            .format(self._input_string_name, self._output_name)
