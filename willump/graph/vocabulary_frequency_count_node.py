from willump.graph.willump_graph_node import WillumpGraphNode
import willump.evaluation.willump_executor as wexec

from typing import List, Tuple

import os
import importlib

class VocabularyFrequencyCountNode(WillumpGraphNode):
    """
    Willump Vocabulary Frequency Count node.  Take in a list of words and an ordered vocabulary
    and return a vector of how many times each word in the vocabulary occurred in the list.
    """
    _input_node: WillumpGraphNode
    _input_string_name: str
    _output_name: str
    _vocabulary_list: List[str]

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_vocabulary_filename: str) -> None:
        self._input_node = input_node
        self._input_string_name = input_node.get_output_name()
        self._output_name = output_name
        with open(input_vocabulary_filename, "r") as vocab_file:
            self._vocabulary_list = vocab_file.read().splitlines()

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return [self._input_node]

    def get_node_type(self) -> str:
        return "string_split"

    def get_aux_data(self) -> Tuple[int, str]:
        """
        Returns a pointer to a Weld dict[[vec[i8], i64] mapping between words and their indices
        in the vocabulary.
        """
        weld_program = "let vocab_dict = dictmerger[vec[i8], i64, +];\n"
        for word_index, word in enumerate(self._vocabulary_list):
            weld_word_list = list(map(lambda char: str(ord(char)) + "c,", word))
            weld_word = "["
            for entry in weld_word_list:
                weld_word += entry
            weld_word += "]"
            word_str = "let vocab_dict = merge(vocab_dict, {WELD_WORD, WORD_INDEX});\n"
            word_str = word_str.replace("WELD_WORD", weld_word)
            word_str = word_str.replace("WORD_INDEX", str(word_index) + "L")
            weld_program += word_str
        weld_program += "result(vocab_dict)"
        print(weld_program)
        module_name = wexec.compile_weld_program(weld_program, {}, "vocabulary_to_dict_driver")
        vocab_to_dict = importlib.import_module(module_name)
        weld_output = vocab_to_dict.caller_func()
        print(weld_output)
        return 0, "dict[[vec[i8], i64]"

    def get_node_weld(self) -> str:
        weld_program = \
            """
            let OUTPUT_NAME = INPUT_NAME;
            """
        weld_program = weld_program.replace("INPUT_NAME", self._input_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "String Split node for input {0} output {1}\n"\
            .format(self._input_string_name, self._output_name)
