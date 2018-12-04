from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
import willump.evaluation.willump_executor as wexec

from weld.types import *

from typing import List, Tuple, Mapping
import importlib


class VocabularyFrequencyCountNode(WillumpGraphNode):
    """
    Willump Vocabulary Frequency Count node.  Take in a list of words and an ordered vocabulary
    and return a vector of how many times each word in the vocabulary occurred in the list.
    """
    _input_nodes: List[WillumpGraphNode]
    _input_string_name: str
    _output_name: str
    _vocab_dict_name: str
    _big_zeros_vector_name: str
    _vocab_size: int

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_vocab_dict: Mapping[str, int], aux_data: List[Tuple[int, WeldType]]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name
        vocabulary_list = sorted(input_vocab_dict.keys(), key=lambda x: input_vocab_dict[x])
        self._vocab_size = len(vocabulary_list)
        self._vocab_dict_name = "AUX_DATA_{0}".format(len(aux_data))
        self._big_zeros_vector_name = "AUX_DATA_{0}".format(len(aux_data) + 1)
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._vocab_dict_name))
        self._input_nodes.append(WillumpInputNode(self._big_zeros_vector_name))
        for entry in self._process_aux_data(vocabulary_list):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "vocab_freq_count"

    @staticmethod
    def _process_aux_data(vocabulary_list) -> List[Tuple[int, WeldType]]:
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
        vocab_dict, zeros_vec = vocab_to_dict.caller_func(vocabulary_list)
        return [(vocab_dict, WeldDict(WeldVec(WeldChar()), WeldLong())),
                (zeros_vec, WeldVec(WeldLong()))]

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let OUTPUT_NAME: vec[i64] = result(for(INPUT_NAME,
                vecmerger[i64, +](BIG_ZEROS_VECTOR),
                | bs: vecmerger[i64, +], i: i64, word: vec[i8] |
                    let exists_and_key = optlookup(VOCAB_DICT_NAME, word);
                    if(exists_and_key.$0,
                        merge(bs, {exists_and_key.$1, 1L}),
                        bs
                    )
            ));
            """
        weld_program = weld_program.replace("VOCAB_SIZE", "{0}L".format(self._vocab_size))
        weld_program = weld_program.replace("VOCAB_DICT_NAME", self._vocab_dict_name)
        weld_program = weld_program.replace("BIG_ZEROS_VECTOR", self._big_zeros_vector_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Vocabulary frequency count node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
