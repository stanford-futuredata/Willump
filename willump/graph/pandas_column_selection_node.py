from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List, Tuple, Mapping


class PandasColumnSelectionNode(WillumpGraphNode):
    """
    Returns a dataframe containing a selection of columns from another dataframe.
    """
    selected_columns: List[str]

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 input_type: WeldType, selected_columns: List[str]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_name
        self._output_name = output_name
        self._input_nodes = [input_node]
        self._input_type = input_type
        self.selected_columns = selected_columns
        self._model_type = None
        self._model_parameters = None
        self._model_index_map = None
        self._input_names = [input_name]

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def push_model(self, model_type: str, model_parameters: Tuple, model_index_map: Mapping[str, int]):
        self._model_type = model_type
        self._model_parameters = model_parameters
        self._model_index_map = model_index_map

    def get_node_weld(self) -> str:
        assert(isinstance(self._input_type, WeldPandas))
        if self._model_type is None:
            selection_string = ""
            input_columns = self._input_type.column_names
            for column in self.selected_columns:
                selection_string += "INPUT_NAME.$%d," % input_columns.index(column)
            weld_program = "let OUTPUT_NAME = {%s};" % selection_string
        else:
            # TODO:  Handle unbatched case.
            assert(self._model_type == "linear")
            weights_name, = self._model_parameters
            sum_string = ""
            input_columns = self._input_type.column_names
            for column in self.selected_columns:
                sum_string += "lookup(WEIGHTS_NAME, %dL) * f64(lookup(INPUT_NAME.$%d, result_i))+" % (self._model_index_map[column], input_columns.index(column))
            sum_string = sum_string[:-1]
            weld_program = \
                """
                let OUTPUT_NAME: vec[f64] = result(for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                    appender[f64],
                    | results: appender[f64], iter_num: i64, result_i: i64 |
                        let sum: f64 = SUM_STRING;
                        merge(results, sum)
                ));
                """
            weld_program = weld_program.replace("SUM_STRING", sum_string)
            weld_program = weld_program.replace("WEIGHTS_NAME", weights_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Pandas column selection node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
