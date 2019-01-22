from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode

from weld.types import *

from typing import List, Tuple, Mapping, Optional
import importlib


class ArrayTfIdfNode(WillumpGraphNode):
    """
    Willump Array TfIdf Vectorizer node.  Take in a list of strings and an ordered vocabulary
    and return vectors of the normalized term frequency of each word in the vocabulary times their inverse
    document frequency.
    """
    # Public variables.
    output_width: int

    # Protected Model-Pushing Variables
    _model_type: Optional[str] = None
    _model_parameters: Tuple
    _start_index: int

    def __init__(self, input_node: WillumpGraphNode, output_name: str,
                 input_vocab_dict: Mapping[str, int], input_idf_vector, aux_data: List[Tuple[int, WeldType]],
                 ngram_range: Tuple[int, int]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_node.get_output_name()
        self._output_name = output_name
        vocabulary_list = sorted(input_vocab_dict.keys(), key=lambda x: input_vocab_dict[x])
        self._vocab_size = len(vocabulary_list)
        self.output_width = self._vocab_size
        self._vocab_dict_name = "AUX_DATA_{0}".format(len(aux_data))
        self._idf_vector_name = "AUX_DATA_{0}".format(len(aux_data) + 1)
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._vocab_dict_name))
        self._input_nodes.append(WillumpInputNode(self._idf_vector_name))
        self._min_gram, self._max_gram = ngram_range
        for entry in self._process_aux_data(vocabulary_list, input_idf_vector):
            aux_data.append(entry)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_node_type(self) -> str:
        return "array count vectorizer"

    def _process_aux_data(self, vocabulary_list, idf_vector) -> List[Tuple[int, WeldType]]:
        """
        Returns a pointer to a Weld dict[[vec[i8], i64] mapping between words and their indices
        in the vocabulary and a vec[f64] idf vector.
        """
        weld_program = \
            """
            result(for(_inp0,
                dictmerger[vec[i8], i64, +],
                | bs: dictmerger[vec[i8], i64, +], i: i64, word: vec[i8] |
                    merge(bs, {word, i})
            ))
            """
        import willump.evaluation.willump_executor as wexec
        module_name = wexec.compile_weld_program(weld_program,
                                                 {"__willump_arg0": WeldVec(WeldVec(WeldChar())),
                                                  "__willump_arg1": WeldVec(WeldDouble())},
                                                 base_filename="tfidf_driver")
        vocab_to_dict = importlib.import_module(module_name)
        vocab_dict, idf_pointer = vocab_to_dict.caller_func(vocabulary_list, idf_vector)
        return [(vocab_dict, WeldDict(WeldVec(WeldChar()), WeldLong())), (idf_pointer, WeldVec(WeldDouble()))]

    def push_model(self, model_type: str, model_parameters: Tuple, start_index) -> None:
        self._model_type = model_type
        self._model_parameters = model_parameters
        self._start_index = start_index

    def get_node_weld(self) -> str:
        if self._model_type is None:
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
                let normalization_vec: vec[f64] = map(vec_dict_vecs,
                    | vec_dict: vec[{i64, i64}] |
                    let squared_sum: f64 = result(for(vec_dict,
                        merger[f64, +],
                        | bs: merger[f64, +], i: i64, x: {i64, i64} |
                        let idf_term: f64 = lookup(IDF_VEC_NAME, x.$0);
                        merge(bs, f64(x.$1)*  idf_term * f64(x.$1) * idf_term)
                    ));
                    let normalization_term = 1.0 / sqrt(squared_sum);
                    normalization_term
                );
                let out_struct: {appender[i64], appender[i64], appender[f64]} = for(vec_dict_vecs,
                    {appender[i64], appender[i64], appender[f64]}, # Row number, index number, frequency
                    | bs: {appender[i64], appender[i64], appender[f64]}, i: i64, x: vec[{i64, i64}] |
                        let normalization_term: f64 = lookup(normalization_vec, i);
                        for(x,
                            bs,
                            | bs2: {appender[i64], appender[i64], appender[f64]}, i2: i64, x2: {i64, i64} |
                                let idf_term: f64 = lookup(IDF_VEC_NAME, x2.$0);
                                {merge(bs2.$0, i), merge(bs2.$1, x2.$0),
                                  merge(bs2.$2, f64(x2.$1) * normalization_term * idf_term)}
                        )
                );
                let ros: {vec[i64], vec[i64], vec[f64]} =
                    {result(out_struct.$0), result(out_struct.$1), result(out_struct.$2)};

                let OUTPUT_NAME: {vec[i64], vec[i64], vec[f64], i64, i64} = {ros.$0, ros.$1, ros.$2, 
                    len(INPUT_NAME), VOCAB_SIZEL}; # Row numbers, index numbers, frequencies, height, width
                """
        else:
            assert (self._model_type == "linear")
            weights_data_name, = self._model_parameters
            weld_program = \
                """
                let OUTPUT_NAME: vec[f64] = result(for(INPUT_NAME,
                    appender[f64],
                    | results:  appender[f64], i_out: i64, string: vec[i8] |
                        let string_len: i64 = len(string);
                        let lin_reg_sum_norm_sum: {merger[f64, +], merger[f64, +]} = for(string,
                            {merger[f64, +], merger[f64, +]},
                            | count_sums: {merger[f64, +], merger[f64, +]}, i_string: i64, char: i8 |
                            for(rangeiter(NGRAM_MINL, NGRAM_MAXL + 1L, 1L),
                                count_sums,
                                | count_sums_inner: {merger[f64, +], merger[f64, +]}, num_iter, iter_value |
                                    if(i_string + iter_value <= string_len,
                                        let word: vec[i8] = slice(string, i_string, iter_value);
                                        let exists_and_key = optlookup(VOCAB_DICT_NAME, word);
                                        if(exists_and_key.$0,
                                            let weight = lookup(WEIGHTS_NAME, START_INDEXl
                                              + exists_and_key.$1);
                                            let idf = lookup(IDF_VEC_NAME, exists_and_key.$1);
                                            {merge(count_sums_inner.$0, weight * idf),
                                            merge(count_sums_inner.$1, idf * idf)},
                                            count_sums_inner
                                        ),
                                        count_sums_inner
                                    )
                            )
                        );
                        merge(results, result(lin_reg_sum_norm_sum.$0) / sqrt(result(lin_reg_sum_norm_sum.$1)))
                ));
                """
            weld_program = weld_program.replace("START_INDEX", str(self._start_index))
            weld_program = weld_program.replace("WEIGHTS_NAME", weights_data_name)
        weld_program = weld_program.replace("VOCAB_DICT_NAME", self._vocab_dict_name)
        weld_program = weld_program.replace("IDF_VEC_NAME", self._idf_vector_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("NGRAM_MIN", str(self._min_gram))
        weld_program = weld_program.replace("NGRAM_MAX", str(self._max_gram))
        weld_program = weld_program.replace("VOCAB_SIZE", str(self._vocab_size))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n" \
            .format(self._input_array_string_name, self._output_name)
