from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
import willump.evaluation.willump_executor as wexec

from weld.types import *

from typing import List, Tuple, Mapping
import importlib


class ArrayCountVectorizerNode(WillumpGraphNode):
    """
    Willump Array Count Vectorizer node.  Take in a list of strings and an ordered vocabulary
    and return vectors of how many times each word in the vocabulary occurred in each string in the list.
    """
    _input_nodes: List[WillumpGraphNode]
    _input_array_string_name: str
    _output_name: str
    _vocab_dict_name: str
    _big_zeros_vector_name: str
    _vocab_size: int
    _min_gram: int
    _max_gram: int

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_vocab_dict: Mapping[str, int], aux_data: List[Tuple[int, WeldType]],
                 ngram_range: Tuple[int, int]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name
        vocabulary_list = sorted(input_vocab_dict.keys(), key=lambda x: input_vocab_dict[x])
        self._vocab_size = len(vocabulary_list)
        self._vocab_dict_name = "AUX_DATA_{0}".format(len(aux_data))
        self._big_zeros_vector_name = "AUX_DATA_{0}".format(len(aux_data) + 1)
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._vocab_dict_name))
        self._input_nodes.append(WillumpInputNode(self._big_zeros_vector_name))
        self._min_gram, self._max_gram = ngram_range
        for entry in self._process_aux_data(vocabulary_list):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

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
            let list_dicts: vec[dict[i64, i64]] = result(for(INPUT_NAME,
                appender[dict[i64, i64]],
                | count_dicts: appender[dict[i64, i64]], i_out: i64, string: vec[i8] |
                    let string_len: i64 = len(string);
                    let string_dict: dict[i64, i64] = result(for(string,
                        dictmerger[i64, i64, +],
                        | count_dict:  dictmerger[i64, i64, +], i_string: i64, char: i8 |
                        for(rangeiter(NGRAM_MINL, NGRAM_MAXL + 1L, 1L),
                            count_dict,
                            | count_dict_inner: dictmerger[i64, i64, +], num_iter, iter_value |
                                if(i_string + iter_value <= string_len,
                                    let word: vec[i8] = slice(string, i_string, iter_value);
                                    let exists_and_key = optlookup(VOCAB_DICT_NAME, word);
                                    if(exists_and_key.$0,
                                        merge(count_dict_inner, {exists_and_key.$1, 1L}),
                                        count_dict_inner
                                    ),
                                    count_dict_inner    
                                )
                        )
                    ));
                    merge(count_dicts, string_dict)
            ));
            let vec_dict_vecs: vec[vec[{i64, i64}]] = map(list_dicts, 
                | string_dict: dict[i64, i64] | 
                    tovec(string_dict)
            );
            let out_struct: {appender[i64], appender[i64], appender[i64]} = for(vec_dict_vecs,
                {appender[i64], appender[i64], appender[i64]}, # Row number, index number, frequency
                | bs: {appender[i64], appender[i64], appender[i64]}, i: i64, x: vec[{i64, i64}] |
                    for(x,
                        bs,
                        | bs2: {appender[i64], appender[i64], appender[i64]}, i2: i64, x2: {i64, i64} |
                            {merge(bs2.$0, i), merge(bs2.$1, x2.$0), merge(bs2.$2, x2.$1)}
                    )
            );
            let ros: {vec[i64], vec[i64], vec[i64]} =
                {result(out_struct.$0), result(out_struct.$1), result(out_struct.$2)};
                    
            let OUTPUT_NAME: {vec[i64], vec[i64], vec[i64], i64} = {ros.$0, ros.$1, ros.$2, len(INPUT_NAME)};
            """
        weld_program = weld_program.replace("VOCAB_DICT_NAME", self._vocab_dict_name)
        weld_program = weld_program.replace("BIG_ZEROS_VECTOR", self._big_zeros_vector_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("NGRAM_MIN", str(self._min_gram))
        weld_program = weld_program.replace("NGRAM_MAX", str(self._max_gram))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n"\
            .format(self._input_array_string_name, self._output_name)
