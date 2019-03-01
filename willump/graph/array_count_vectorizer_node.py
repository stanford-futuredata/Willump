from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode

from weld.types import *

from typing import List, Tuple, Mapping, Optional
import importlib


class ArrayCountVectorizerNode(WillumpGraphNode):
    """
    Willump Array Count Vectorizer node.  Take in a list of strings and an ordered vocabulary
    and return vectors of how many times each word in the vocabulary occurred in each string in the list.
    """
    # Public variables.
    output_width: int

    # Protected Model-Pushing Variables
    _model_type: Optional[str] = None
    _model_parameters: Tuple
    _start_index: int

    # Protected Cascade Variables
    _small_model_output_name: Optional[str] = None

    def __init__(self, input_node: WillumpGraphNode, input_name: str, output_name: str,
                 input_vocab_dict: Mapping[str, int], aux_data: List[Tuple[int, WeldType]],
                 ngram_range: Tuple[int, int]) -> None:
        """
        Initialize the node, appending a new entry to aux_data in the process.
        """
        self._input_array_string_name = input_name
        self._output_name = output_name
        vocabulary_list = sorted(input_vocab_dict.keys(), key=lambda x: input_vocab_dict[x])
        self._vocab_size = len(vocabulary_list)
        self.output_width = self._vocab_size
        self._vocab_dict_name = "AUX_DATA_{0}".format(len(aux_data))
        self._input_nodes = []
        self._input_nodes.append(input_node)
        self._input_nodes.append(WillumpInputNode(self._vocab_dict_name))
        self._input_names = [input_name, self._vocab_dict_name]
        self._min_gram, self._max_gram = ngram_range
        for entry in self._process_aux_data(vocabulary_list):
            aux_data.append(entry)

    def get_cost(self) -> float:
        return 4 * (self._max_gram - self._min_gram + 1)

    def get_in_nodes(self) -> List[WillumpGraphNode]:
        return self._input_nodes

    def get_in_names(self) -> List[str]:
        return self._input_names

    def _process_aux_data(self, vocabulary_list) -> List[Tuple[int, WeldType]]:
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
        import willump.evaluation.willump_executor as wexec
        module_name = wexec.compile_weld_program(weld_program,
                                                 type_map={"input": WeldVec(WeldVec(WeldChar()))},
                                                 input_names=["input"],
                                                 output_names=[],
                                                 base_filename="vocabulary_to_dict_driver")
        vocab_to_dict = importlib.import_module(module_name)
        vocab_dict, _ = vocab_to_dict.caller_func(vocabulary_list)
        return [(vocab_dict, WeldDict(WeldVec(WeldChar()), WeldLong()))]

    def push_model(self, model_type: str, model_parameters: Tuple, start_index) -> None:
        self._model_type = model_type
        self._model_parameters = model_parameters
        self._start_index = start_index

    def push_cascade(self, small_model_output_node: WillumpGraphNode):
        small_model_output_name = small_model_output_node.get_output_names()[0]
        self._small_model_output_name = small_model_output_name
        self._input_nodes.append(small_model_output_node)
        self._input_names.append(small_model_output_name)

    def get_node_weld(self) -> str:
        if self._model_type is None:
            if self._small_model_output_name is None:
                cascade_statement = "true"
            else:
                cascade_statement = "lookup(%s, i_out) == 2c" % self._small_model_output_name
            weld_program = \
                """
                let list_dicts: vec[dict[i64, i64]] = result(for(INPUT_NAME,
                    appender[dict[i64, i64]],
                    | count_dicts: appender[dict[i64, i64]], i_out: i64, string: vec[i8] |
                        if(CASCADE_STATEMENT,
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
                            merge(count_dicts, string_dict),
                            count_dicts
                        )
                ));
                let a = assert(len(list_dicts) >= 0L); # TODO:  Talk to Shoumik about this.
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
                if(a,
                    {result(out_struct.$0), result(out_struct.$1), result(out_struct.$2)},
                    
                    {result(out_struct.$0), result(out_struct.$1), result(out_struct.$2)}
                    );
                        
                let OUTPUT_NAME: {vec[i64], vec[i64], vec[i64], i64, i64} = {ros.$0, ros.$1, ros.$2, 
                    len(INPUT_NAME), VOCAB_SIZEL}; # Row numbers, index numbers, frequencies, height, width
                """
            weld_program = weld_program.replace("CASCADE_STATEMENT", cascade_statement)
        else:
            assert(self._model_type == "linear")
            weights_data_name, = self._model_parameters
            weld_program = \
                """
                let OUTPUT_NAME: vec[f64] = result(for(INPUT_NAME,
                    appender[f64],
                    | results:  appender[f64], i_out: i64, string: vec[i8] |
                        let string_len: i64 = len(string);
                        let lin_reg_sum: f64 = result(for(string,
                            merger[f64, +],
                            | count_sum:  merger[f64, +], i_string: i64, char: i8 |
                            for(rangeiter(NGRAM_MINL, NGRAM_MAXL + 1L, 1L),
                                count_sum,
                                | count_sum_inner: merger[f64, +], num_iter, iter_value |
                                    if(i_string + iter_value <= string_len,
                                        let word: vec[i8] = slice(string, i_string, iter_value);
                                        let exists_and_key = optlookup(VOCAB_DICT_NAME, word);
                                        if(exists_and_key.$0,
                                            merge(count_sum_inner, lookup(WEIGHTS_NAME, START_INDEXl
                                              + exists_and_key.$1)),
                                            count_sum_inner
                                        ),
                                        count_sum_inner
                                    )
                            )
                        ));
                        merge(results, lin_reg_sum)
                ));
                """
            weld_program = weld_program.replace("START_INDEX", str(self._start_index))
            weld_program = weld_program.replace("WEIGHTS_NAME", weights_data_name)
        weld_program = weld_program.replace("VOCAB_DICT_NAME", self._vocab_dict_name)
        weld_program = weld_program.replace("INPUT_NAME", self._input_array_string_name)
        weld_program = weld_program.replace("OUTPUT_NAME", self._output_name)
        weld_program = weld_program.replace("NGRAM_MIN", str(self._min_gram))
        weld_program = weld_program.replace("NGRAM_MAX", str(self._max_gram))
        weld_program = weld_program.replace("VOCAB_SIZE", str(self._vocab_size))
        return weld_program

    def get_output_name(self) -> str:
        return self._output_name

    def get_output_names(self) -> List[str]:
        return [self._output_name]

    def get_output_type(self) -> WeldType:
        return WeldCSR(WeldLong())

    def get_output_types(self) -> List[WeldType]:
        return [WeldCSR(WeldLong())]

    def __repr__(self):
        return "Array count-vectorizer node for input {0} output {1}\n"\
            .format(self._input_array_string_name, self._output_name)
