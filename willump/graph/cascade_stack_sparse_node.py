from willump.graph.willump_graph_node import WillumpGraphNode

from weld.types import *

from typing import List


class CascadeStackSparseNode(WillumpGraphNode):
    """
    Willump Stack Sparse node.  Horizontally stacks multiple sparse matrices.
    """

    def __init__(self, more_important_nodes: List[WillumpGraphNode], more_important_names: List[str],
                 less_important_nodes: List[WillumpGraphNode], less_important_names: List[str],
                 output_name: str, output_type: WeldType, small_model_output_node: WillumpGraphNode,
                 small_model_output_name: str) -> None:
        """
        Initialize the node.
        """
        self._input_nodes = more_important_nodes + less_important_nodes + [small_model_output_node]
        self._output_name = output_name
        assert(isinstance(output_type, WeldCSR))
        self._elem_type = output_type.elemType
        self._output_type = output_type
        self._input_names = more_important_names + less_important_names + [small_model_output_name]
        self._more_important_names = more_important_names
        self._less_important_names = less_important_names
        self._small_model_output_name = small_model_output_name

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def get_node_weld(self) -> str:
        num_rows = ""
        for input_name in (self._more_important_names + self._less_important_names):
            num_rows += "len(%s.$0)+" % input_name
        num_rows = num_rows[:-1]
        weld_program = \
            """
            let num_rows: i64 = NUM_ROWS;
            let stacker = {appender[i64](num_rows), appender[i64](num_rows), appender[ELEM_TYPE](num_rows)};
            let stack_width: i64 = 0L;
            let df_size = len(SMALL_MODEL_OUTPUT);
            let pre_mapping = iterate({0L, 0L, dictmerger[i64, i64, +]},
                | input |
                    let more_important_iter = input.$0;
                    let less_important_iter = input.$1;
                    let output_dict = input.$2;
                    if(more_important_iter == df_size,
                        {{more_important_iter, less_important_iter, output_dict}, false},
                        let small_model_output = lookup(SMALL_MODEL_OUTPUT, more_important_iter);
                        if(small_model_output != 2c,
                            {{more_important_iter + 1L, less_important_iter, output_dict}, true},
                            {{more_important_iter + 1L, less_important_iter + 1L, merge(output_dict, {more_important_iter, less_important_iter})}, true}
                        )
                    )
            );  
            let small_model_output_len = pre_mapping.$1;
            let small_model_output_mapping = result(pre_mapping.$2);
            """
        for more_important_input_name in self._more_important_names:
            weld_program += \
                """
                let stacker = for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                    stacker,
                    | bs, i, x: i64 |
                    let row_number = lookup(INPUT_NAME.$0, x);
                    let mod_row_number_presence = optlookup(small_model_output_mapping, row_number);
                    if(mod_row_number_presence.$0, 
                        {merge(bs.$0,mod_row_number_presence.$1), 
                          merge(bs.$1, lookup(INPUT_NAME.$1, x) + stack_width), 
                          merge(bs.$2, ELEM_TYPE(lookup(INPUT_NAME.$2, x)))
                        },
                        bs
                    )
                );
                let stack_width = stack_width + INPUT_NAME.$4;
                """
            weld_program = weld_program.replace("INPUT_NAME", more_important_input_name)
        for less_important_input_name in self._less_important_names:
            weld_program += \
                """
                let stacker = for(rangeiter(0L, len(INPUT_NAME.$0), 1L),
                    stacker,
                    | bs, i, x: i64 |
                    {merge(bs.$0, lookup(INPUT_NAME.$0, x)), 
                      merge(bs.$1, lookup(INPUT_NAME.$1, x) + stack_width), 
                      merge(bs.$2, ELEM_TYPE(lookup(INPUT_NAME.$2, x)))
                    }
                );
                let stack_height = INPUT_NAME.$3;
                let stack_width = stack_width + INPUT_NAME.$4;
                """
            weld_program = weld_program.replace("INPUT_NAME", less_important_input_name)
        weld_program += \
            """
            let stack_height: i64 = small_model_output_len;
            let OUTPUT_NAME: {vec[i64], vec[i64], vec[ELEM_TYPE], i64, i64} = {result(stacker.$0), 
                result(stacker.$1), result(stacker.$2), stack_height, stack_width};
            """
        weld_program = weld_program.replace("NUM_ROWS", num_rows)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("SMALL_MODEL_OUTPUT", self._small_model_output_name)
        weld_program = weld_program.replace("ELEM_TYPE", str(self._elem_type))
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
        return "Stack sparse node for input {0} output {1}\n" \
            .format(self._input_names, self._output_name)
