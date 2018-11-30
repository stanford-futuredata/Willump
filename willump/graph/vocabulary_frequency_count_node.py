from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
import willump.evaluation.willump_executor as wexec

from weld.types import *

from typing import List, Tuple
import importlib


class VocabularyFrequencyCountNode(WillumpGraphNode):
    """
    Willump Vocabulary Frequency Count node.  Take in a list of words and an ordered vocabulary
    and return a vector of how many times each word in the vocabulary occurred in the list.
    """
    _input_nodes: List[WillumpGraphNode]
    _input_string_name: str
    _output_name: str
    _aux_data_name: str
    _vocab_size: int

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_vocabulary_filename: str, aux_data: List[Tuple[int, str]]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name
        with open(input_vocabulary_filename, "r") as vocab_file:
            vocabulary_list = vocab_file.read().splitlines()
        self._vocab_size = len(vocabulary_list)
        self._aux_data_name = "AUX_DATA_{0}".format(len(aux_data))
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._aux_data_name))
        aux_data.append(self._process_aux_data(vocabulary_list))

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "string_split"

    @staticmethod
    def _process_aux_data(vocabulary_list) -> Tuple[int, str]:
        """
        Returns a pointer to a Weld dict[[vec[i8], i64] mapping between words and their indices
        in the vocabulary.
        """
        weld_program = \
        """
        result(for(_inp0,
            dictmerger[vec[i8], i64, +],
            | bs: dictmerger[vec[i8], i64, +], i: i64, word: vec[i8] |
                merge(bs, {word, i})
        ))
        """
        module_name = wexec.compile_weld_program(weld_program,
                                                 {"__willump_arg0": WeldVec(WeldVec(WeldChar()))},
                                                 base_filename="vocabulary_to_dict_driver")
        vocab_to_dict = importlib.import_module(module_name)
        weld_output = vocab_to_dict.caller_func(vocabulary_list)
        return weld_output, "dict[vec[i8], i64]"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let index_frequency_map: dict[i64,i64] = result(for(INPUT_NAME,
                dictmerger[i64,i64,+],
                | bs: dictmerger[i64,i64,+], i: i64, word: vec[i8]|
                    let exists_and_key = optlookup(AUX_INPUT_NAME, word);
                    if(exists_and_key.$0,
                        merge(bs, {exists_and_key.$1, 1L}),
                        bs
                    )
            ));
            let frequency_vector_struct = iterate({appender[i64], 0L},
                | appender_index: {appender[i64], i64} |
                    if(appender_index.$1 == VOCAB_SIZE,
                        {appender_index, false},
                        let exists_and_key = optlookup(index_frequency_map, appender_index.$1);
                        if(exists_and_key.$0,
                            {{merge(appender_index.$0, exists_and_key.$1),
                                appender_index.$1 + 1L}, true},
                            {{merge(appender_index.$0, 0L), appender_index.$1 + 1L}, true}
                        )
                    )
            );
            let OUTPUT_NAME = result(frequency_vector_struct.$0);
            """
        weld_program = weld_program.replace("VOCAB_SIZE", "{0}L".format(self._vocab_size))
        weld_program = weld_program.replace("AUX_INPUT_NAME", self._aux_data_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "String Split node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
