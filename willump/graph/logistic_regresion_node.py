from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
import willump.evaluation.willump_executor as wexec

from weld.types import *

from typing import List, Tuple, Mapping
import importlib
import numpy


class LogisticRegressionNode(WillumpGraphNode):
    """
    Willump Logistic Regression node.  Evaluates a logistic regression model on an input.

    TODO:  Allow multi-class classification, not just binary.
    """
    _input_nodes: List[WillumpGraphNode]
    _input_string_name: str
    _output_name: str
    _weights_data_name: str
    _intercept_data_name: str

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 logit_weights, logit_intercept, aux_data: List[Tuple[int, str]]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name
        self._weights_data_name = "AUX_DATA_{0}".format(len(aux_data))
        self._intercept_data_name = "AUX_DATA_{0}".format(len(aux_data) + 1)
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._weights_data_name))
        self._input_nodes.append(WillumpInputNode(self._intercept_data_name))
        for entry in self._process_aux_data(logit_weights, logit_intercept):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "logistic_regression"

    @staticmethod
    def _process_aux_data(logit_weights, logit_intercept) -> List[Tuple[int, str]]:
        """
        Returns pointers to Weld vectors containing the logistic regression model weights
        and intercept.
        """
        weld_program = \
            """
            true
            """
        module_name = wexec.compile_weld_program(weld_program, {},
                                                 base_filename="encode_logistic_regression_model")
        encode_logit_model = importlib.import_module(module_name)
        weld_weights, weld_intercept = encode_logit_model.caller_func(logit_weights[0], logit_intercept)
        return [(weld_weights, "vec[f64]"), (weld_intercept, "vec[f64]")]

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let output_float: f64 = result(for(zip(INPUT_NAME, WEIGHTS_NAME),
                merger[f64, +],
                | bs: merger[f64, +], i: i64, inp_weight: {i64, f64} |
                    merge(bs, f64(inp_weight.$0) * inp_weight.$1)
            ));
            let output_float: f64 = output_float + lookup(INTERCEPT_NAME, 0L);
            let output_int: i64 = select(output_float > 0.0, 1L, 0L);
            let OUTPUT_NAME = result(merge(appender[i64], output_int));
            """
        weld_program = weld_program.replace("WEIGHTS_NAME", self._weights_data_name)
        weld_program = weld_program.replace("INTERCEPT_NAME", self._intercept_data_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Logistic regression node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
