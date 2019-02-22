from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List, Tuple, Mapping


class PandasColumnSelectionNode(WillumpGraphNode):
    """
    Returns a dataframe containing a selection of columns from other dataframes.
    """
    selected_columns: List[str]

    def __init__(self, input_nodes: List[WillumpGraphNode], input_names: List[str], output_name: str,
                 input_types: List[WeldType], output_type: WeldType, selected_columns: List[str]) -> None:
        """
        Initialize the node.
        """
        self._output_name = output_name
        self._input_nodes = input_nodes
        self._input_types = input_types
        self.selected_columns = selected_columns
        self._model_type = None
        self._model_parameters = None
        self._model_index_map = None
        self._input_names = input_names
        self._output_type = output_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def push_model(self, model_type: str, model_parameters: Tuple, model_index_map: Mapping[str, int]):
        self._model_type = model_type
        self._model_parameters = model_parameters
        self._model_index_map = model_index_map

    def get_node_weld(self) -> str:
        if self._model_type is None:
            if isinstance(self._output_type, WeldPandas):
                assert (all(isinstance(input_type, WeldPandas) for input_type in self._input_types))
                selection_string = ""
                for column in self.selected_columns:
                    for input_name, input_type in zip(self._input_names, self._input_types):
                        input_columns = input_type.column_names
                        if column in input_columns:
                            selection_string += "%s.$%d," % (input_name, input_columns.index(column))
                            break
                weld_program = "let OUTPUT_NAME = {%s};" % selection_string
            else:
                assert (isinstance(self._output_type, WeldSeriesPandas))
                assert (all(isinstance(input_type, WeldSeriesPandas) for input_type in self._input_types))
                weld_program = "let pre_output = appender[%s];\n" % str(self._output_type.elemType)
                for column in self.selected_columns:
                    for input_name, input_type in zip(self._input_names, self._input_types):
                        input_columns = input_type.column_names
                        if column in input_columns:
                            col_index = input_columns.index(column)
                            weld_program += "let pre_output = merge(pre_output, lookup(%s, %dL));\n" % (input_name, col_index)
                weld_program += "let OUTPUT_NAME = result(pre_output);\n"
        else:
            assert(self._model_type == "linear")
            assert(len(self._input_names) == 1)
            assert(isinstance(self._output_type, WeldPandas))
            weights_name, = self._model_parameters
            sum_string = ""
            input_columns = self._input_types[0].column_names
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
            weld_program = weld_program.replace("INPUT_NAME", self._input_names[0])
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def get_output_type(self) -> WeldType:
        return self._output_type

    def get_output_types(self) -> List[WeldType]:
        return [self._output_type]

    def __repr__(self):
        return "Pandas column selection node for inputs {0} output {1}\n" \
            .format(self._input_names, self._output_name)
