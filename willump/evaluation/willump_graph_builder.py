import ast

from willump import *
from willump.willump_utilities import *
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_lower_node import StringLowerNode
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.array_binop_node import ArrayBinopNode
from willump.graph.hash_join_node import WillumpHashJoinNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.linear_training_node import LinearTrainingNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from willump.graph.trees_training_node import TreesTrainingNode
from willump.graph.trees_model_node import TreesModelNode
from willump.graph.pandas_series_to_dataframe_node import PandasSeriesToDataFrameNode
from willump.graph.pandas_series_concatenation_node import PandasSeriesConcatenationNode
from willump.graph.identity_node import IdentityNode
from willump.graph.reshape_node import ReshapeNode

from typing import MutableMapping, List, Tuple, Optional, Mapping
from weld.types import *


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from the Python AST for a FunctionDef.

    # TODO:  Compile nodes appearing in control flow.
    """
    willump_graph: WillumpGraph
    # A list of all argument names in the order they are passed in.
    arg_list: List[str]
    # A map from the names of variables to the nodes that generate them.
    _node_dict: MutableMapping[str, WillumpGraphNode]
    # A map from variables to their Weld types.
    _type_map: MutableMapping[str, WeldType]
    # A set of static variable values saved from Python execution.
    _static_vars: Mapping[str, object]
    # Saved data structures required by some nodes.
    aux_data: List[Tuple[int, WeldType]]
    # Temporary variable counter
    _temp_var_counter: int
    # Asynchronous Python functions to execute on parallel threads if possible.
    _async_funcs: List[str]

    def __init__(self, type_map: MutableMapping[str, WeldType],
                 static_vars: Mapping[str, object], async_funcs: List[str], cached_funcs: List[str]) -> None:
        self._node_dict = {}
        self._type_map = type_map
        self._static_vars = static_vars
        self.arg_list = []
        self.aux_data = []
        self._temp_var_counter = 0
        self._async_funcs = async_funcs
        self._cached_funcs = cached_funcs

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Begin processing of a function.  Create input nodes for function arguments.
        """
        for arg in node.args.args:
            arg_name: str = self.get_store_name(arg.arg, node.lineno)
            input_node: WillumpInputNode = WillumpInputNode(arg_name)
            self._node_dict[arg_name] = input_node
            self.arg_list.append(arg_name)
        for entry in node.body:
            if isinstance(entry, ast.Assign):
                output_var_name, assignment_node = self.analyze_Assign(entry)
                self._node_dict[output_var_name] = assignment_node
            elif isinstance(entry, ast.Return):
                self.analyze_Return(entry)
            else:
                output_names, py_node = self._create_py_node(entry)
                for output_name in output_names:
                    self._node_dict[output_name] = py_node

    def analyze_Assign(self, node: ast.Assign) -> \
            Tuple[str, WillumpGraphNode]:
        """
        Process an assignment AST node into either a Willump Weld node or Willump Python node that
        defines the variable being assigned.
        """

        def create_single_output_py_node(in_node, is_async_func=False, is_cached_func=False,
                                         does_not_modify_data=False):
            output_names, py_node = self._create_py_node(in_node, is_async_func=is_async_func,
                                                         is_cached_node=is_cached_func,
                                                         does_not_modify_data=does_not_modify_data)
            assert (len(output_names) == 1)
            return output_names[0], py_node

        assert (len(node.targets) == 1)  # Assume assignment to only one variable.
        target: ast.expr = node.targets[0]
        output_var_name_base = self.get_assignment_target_name(target)
        if output_var_name_base is None:
            return create_single_output_py_node(node)
        output_var_name = self.get_store_name(output_var_name_base, target.lineno)
        output_type = self._type_map[output_var_name]
        value: ast.expr = node.value
        if isinstance(value, ast.BinOp):
            if not isinstance(output_type, WeldVec):
                return create_single_output_py_node(node)
            if isinstance(target, ast.Subscript):
                return create_single_output_py_node(node)
            if isinstance(value.left, ast.Name) and value.left.id in self._node_dict:
                left_name: str = self.get_load_name(value.left.id, value.lineno, self._type_map)
                left_node: WillumpGraphNode = self._node_dict[left_name]
            else:
                # TODO:  Type the new node properly.
                left_name, left_node = self._create_temp_var_from_node(value.left, output_type)
                self._node_dict[left_name] = left_node
            if isinstance(value.right, ast.Name) and value.right.id in self._node_dict:
                right_name: str = self.get_load_name(value.right.id, value.lineno, self._type_map)
                right_node: WillumpGraphNode = self._node_dict[right_name]
            else:
                # TODO:  Type the new node properly.
                right_name, right_node = self._create_temp_var_from_node(value.right, output_type)
                self._node_dict[right_name] = right_node
            if isinstance(value.op, ast.Add):
                array_binop_node = ArrayBinopNode(left_node, right_node, left_name, right_name, output_var_name,
                                                  output_type, "+")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Sub):
                array_binop_node = ArrayBinopNode(left_node, right_node, left_name, right_name, output_var_name,
                                                  output_type, "-")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Mult):
                array_binop_node = ArrayBinopNode(left_node, right_node, left_name, right_name, output_var_name,
                                                  output_type, "*")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Div):
                array_binop_node = ArrayBinopNode(left_node, right_node, left_name, right_name, output_var_name,
                                                  output_type, "/")
                return output_var_name, array_binop_node
            else:
                return create_single_output_py_node(node)
        elif isinstance(value, ast.Subscript):
            if isinstance(value.slice, ast.Index) and isinstance(value.slice.value, ast.Name) and \
                    isinstance(value.value, ast.Name):
                input_var_name = self.get_load_name(value.value.id, value.lineno, self._type_map)
                column_names = self._static_vars[WILLUMP_SUBSCRIPT_INDEX_NAME + str(value.lineno)]
                input_var_type = self._type_map[input_var_name]
                output_type = self._type_map[output_var_name]
                if (isinstance(input_var_type, WeldPandas) or isinstance(input_var_type, WeldSeriesPandas))\
                        and isinstance(column_names, list):
                    pandas_column_selection_node = \
                        PandasColumnSelectionNode(input_nodes=[self._node_dict[input_var_name]],
                                                  input_names=[input_var_name],
                                                  output_name=output_var_name,
                                                  input_types=[input_var_type],
                                                  selected_columns=column_names,
                                                  output_type=output_type)
                    return output_var_name, pandas_column_selection_node
                else:
                    return create_single_output_py_node(node)
            else:
                return create_single_output_py_node(node)
        elif isinstance(value, ast.Call):
            called_function: Optional[str] = self._get_function_name(value)
            if called_function is None:
                return create_single_output_py_node(node)
            # Applying a function to a Pandas column.
            elif "apply" in called_function:
                if isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Subscript) \
                        and isinstance(value.func.value.value, ast.Name) and isinstance(value.func.value.slice.value,
                                                                                        ast.Str):
                    input_df_name = self.get_load_name(value.func.value.value.id, value.lineno, self._type_map)
                    input_df_node = self._node_dict[input_df_name]
                    input_df_type = self._type_map[input_df_name]
                    input_col = value.func.value.slice.value.s
                    if isinstance(value.args[0], ast.Attribute) and value.args[0].attr == "lower" and \
                            isinstance(target, ast.Subscript) and isinstance(target.slice.value, ast.Str):
                        output_col = target.slice.value.s
                        string_lower_node: StringLowerNode = StringLowerNode(input_node=input_df_node,
                                                                             input_name=input_df_name,
                                                                             input_type=input_df_type,
                                                                             input_col=input_col,
                                                                             output_name=output_var_name,
                                                                             output_type=self._type_map[
                                                                                 output_var_name],
                                                                             output_col=output_col)
                        return output_var_name, string_lower_node
                    else:
                        return create_single_output_py_node(node)
                else:
                    return create_single_output_py_node(node)
            elif ".predict" in called_function:
                if WILLUMP_LINEAR_REGRESSION_WEIGHTS in self._static_vars:
                    logit_input_var: str = self.get_load_name(value.args[0].id, value.lineno, self._type_map)
                    logit_weights = self._static_vars[WILLUMP_LINEAR_REGRESSION_WEIGHTS]
                    logit_intercept = self._static_vars[WILLUMP_LINEAR_REGRESSION_INTERCEPT]
                    logit_input_node: WillumpGraphNode = self._node_dict[logit_input_var]
                    logit_node: LinearRegressionNode = LinearRegressionNode(
                        input_node=logit_input_node,
                        input_name=logit_input_var,
                        input_type=self._type_map[logit_input_var],
                        output_name=output_var_name,
                        output_type=self._type_map[output_var_name],
                        logit_weights=logit_weights,
                        logit_intercept=logit_intercept, aux_data=self.aux_data
                    )
                    return output_var_name, logit_node
                elif WILLUMP_TREES_FEATURE_IMPORTANCES in self._static_vars:
                    trees_input_var: str = self.get_load_name(value.args[0].id, value.lineno, self._type_map)
                    trees_input_node: WillumpGraphNode = self._node_dict[trees_input_var]
                    model_name = value.func.value.id
                    feature_importances = self._static_vars[WILLUMP_TREES_FEATURE_IMPORTANCES]
                    trees_node = TreesModelNode(input_node=trees_input_node, input_name=trees_input_var,
                                                output_name=output_var_name, model_name=model_name,
                                                input_width=len(feature_importances),
                                                output_type=self._type_map[output_var_name])
                    return output_var_name, trees_node
                else:
                    return create_single_output_py_node(node)
            elif ".transform" in called_function:
                lineno = str(node.lineno)
                if WILLUMP_COUNT_VECTORIZER_ANALYZER + lineno not in self._static_vars or\
                    WILLUMP_COUNT_VECTORIZER_VOCAB + lineno not in self._static_vars or \
                        self._static_vars[WILLUMP_COUNT_VECTORIZER_LOWERCASE + lineno] is True:
                    return create_single_output_py_node(node)
                vectorizer_input_var: str = self.get_load_name(value.args[0].id, node.lineno, self._type_map)
                analyzer: str = self._static_vars[WILLUMP_COUNT_VECTORIZER_ANALYZER + lineno]
                vocab_dict: Mapping[str, int] = self._static_vars[WILLUMP_COUNT_VECTORIZER_VOCAB + lineno]
                vectorizer_input_node: WillumpGraphNode = self._node_dict[vectorizer_input_var]
                vectorizer_ngram_range: Tuple[int, int] = \
                    self._static_vars[WILLUMP_COUNT_VECTORIZER_NGRAM_RANGE + lineno]
                if analyzer == "char":
                    preprocessed_input_prefix = "char__preprocessed__"
                else:
                    assert (analyzer == "word")
                    preprocessed_input_prefix = "word__preprocessed__"
                try:
                    preprocessed_input_name = self.get_load_name(preprocessed_input_prefix + vectorizer_input_var,
                                                                 node.lineno, self._type_map)
                except UntypedVariableException:
                    preprocessed_input_name = self.get_store_name(preprocessed_input_prefix + vectorizer_input_var,
                                                                  node.lineno)
                if preprocessed_input_name in self._node_dict:
                    input_preprocessing_node = self._node_dict[preprocessed_input_name]
                else:
                    if analyzer == "char":
                        input_preprocessing_python = "%s = list(map(lambda x: re.sub(r'\s\s+', ' ', x), %s))" \
                                                     % (strip_linenos_from_var(preprocessed_input_name),
                                                        strip_linenos_from_var(vectorizer_input_var))
                    else:
                        input_preprocessing_python = "%s = list(map(lambda x: re.sub(r'\W+', ' ', x), %s))" \
                                                     % (strip_linenos_from_var(preprocessed_input_name),
                                                        strip_linenos_from_var(vectorizer_input_var))
                    input_preprocessing_ast: ast.Module = ast.parse(input_preprocessing_python)
                    input_preprocessing_node: WillumpPythonNode = WillumpPythonNode(
                        python_ast=input_preprocessing_ast.body[0],
                        input_names=[vectorizer_input_var],
                        output_names=[preprocessed_input_name],
                        output_types=[self._type_map[vectorizer_input_var]],
                        in_nodes=[vectorizer_input_node])
                    self._type_map[preprocessed_input_name] = self._type_map[vectorizer_input_var]
                    self._node_dict[preprocessed_input_name] = input_preprocessing_node
                if WILLUMP_TFIDF_IDF_VECTOR + lineno in self._static_vars:
                    idf_vector = self._static_vars[WILLUMP_TFIDF_IDF_VECTOR + lineno]
                    tfidf_node: ArrayTfIdfNode = ArrayTfIdfNode(
                        input_node=input_preprocessing_node,
                        input_name=preprocessed_input_name,
                        output_name=output_var_name,
                        input_idf_vector=idf_vector,
                        input_vocab_dict=vocab_dict, aux_data=self.aux_data,
                        analyzer=analyzer,
                        ngram_range=vectorizer_ngram_range
                    )
                    return output_var_name, tfidf_node
                else:
                    assert (analyzer == "char")  # TODO:  Get other analyzers working.
                    array_cv_node: ArrayCountVectorizerNode = ArrayCountVectorizerNode(
                        input_node=input_preprocessing_node,
                        input_name=preprocessed_input_name,
                        output_name=output_var_name,
                        input_vocab_dict=vocab_dict, aux_data=self.aux_data,
                        ngram_range=vectorizer_ngram_range
                    )
                    return output_var_name, array_cv_node
            elif ".merge" in called_function:
                if self._static_vars[WILLUMP_JOIN_HOW + str(node.lineno)] is not 'left':
                    return create_single_output_py_node(node)
                join_col: str = self._static_vars[WILLUMP_JOIN_COL + str(node.lineno)]
                right_df = self._static_vars[WILLUMP_JOIN_RIGHT_DATAFRAME + str(node.lineno)]
                left_df_input_var: str = self.get_load_name(value.func.value.id, value.lineno, self._type_map)
                left_df_input_node = self._node_dict[left_df_input_var]
                willump_hash_join_node = WillumpHashJoinNode(input_node=left_df_input_node,
                                                             input_name=left_df_input_var,
                                                             output_name=output_var_name,
                                                             left_input_type=self._type_map[left_df_input_var],
                                                             join_col_name=join_col,
                                                             right_dataframe=right_df, aux_data=self.aux_data)
                return output_var_name, willump_hash_join_node
            elif "sparse.hstack" in called_function:
                if not isinstance(value.args[0], ast.List) or \
                        not all(isinstance(ele, ast.Name) for ele in value.args[0].elts):
                    return create_single_output_py_node(node)
                input_names = [self.get_load_name(arg_node.id, arg_node.lineno, self._type_map) for arg_node in
                               value.args[0].elts]
                input_nodes = [self._node_dict[input_name] for input_name in input_names]
                output_type = self._type_map[output_var_name]
                stack_sparse_node = StackSparseNode(input_nodes=input_nodes,
                                                    input_names=input_names,
                                                    output_name=output_var_name,
                                                    output_type=output_type)
                return output_var_name, stack_sparse_node
            elif ".fit" in called_function:
                if isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) and \
                        len(value.args) == 2 and isinstance(value.args[0], ast.Name) and isinstance(value.args[1],
                                                                                                    ast.Name):
                    x_name = self.get_load_name(value.args[0].id, value.lineno, self._type_map)
                    model_name = self.get_load_name(value.func.value.id, value.lineno, self._type_map)
                    y_name = self.get_load_name(value.args[1].id, value.lineno, self._type_map)
                    x_node, model_node, y_node = self._node_dict[x_name], self._node_dict[model_name], self._node_dict[
                        y_name]
                    if WILLUMP_LINEAR_REGRESSION_WEIGHTS in self._static_vars:
                        input_weights = self._static_vars[WILLUMP_LINEAR_REGRESSION_WEIGHTS]
                        input_intercept = self._static_vars[WILLUMP_LINEAR_REGRESSION_INTERCEPT]
                        training_node = LinearTrainingNode(python_ast=node,
                                                           in_nodes=[x_node, model_node, y_node],
                                                           input_names=[x_name, model_name, y_name],
                                                           output_names=[output_var_name],
                                                           input_weights=input_weights,
                                                           input_intercept=input_intercept)
                        return output_var_name, training_node
                    elif WILLUMP_TREES_FEATURE_IMPORTANCES in self._static_vars:
                        feature_importances = self._static_vars[WILLUMP_TREES_FEATURE_IMPORTANCES]
                        training_node = TreesTrainingNode(python_ast=node,
                                                          in_nodes=[x_node, model_node, y_node],
                                                          input_names=[x_name, model_name, y_name],
                                                          output_names=[output_var_name],
                                                          feature_importances=feature_importances)
                        return output_var_name, training_node
                    else:
                        return create_single_output_py_node(node)
                else:
                    return create_single_output_py_node(node)
            elif ".fillna" in called_function and isinstance(value.func, ast.Attribute) and \
                    isinstance(value.func.value, ast.Name):
                input_name = self.get_load_name(value.func.value.id, value.lineno, self._type_map)
                input_node = self._node_dict[input_name]
                if isinstance(input_node, WillumpHashJoinNode) or isinstance(input_node, PandasColumnSelectionNode):
                    output_type = self._type_map[output_var_name]
                    # NaNs within Hash Join nodes are handled by the nodes.
                    identity_node = IdentityNode(input_node=input_node, input_name=input_name,
                                                 output_name=output_var_name, output_type=output_type)
                    return output_var_name, identity_node
                else:
                    return create_single_output_py_node(node)
            elif ".concat" in called_function and len(value.args) == 1 and isinstance(value.args[0], ast.List) \
                    and all(isinstance(arg_node, ast.Name) for arg_node in value.args[0].elts):
                arg_names: List[str] = [self.get_load_name(arg_node.id, value.lineno, self._type_map)
                                        for arg_node in value.args[0].elts]
                if all(isinstance(self._type_map[arg_name], WeldSeriesPandas) for arg_name in arg_names):
                    input_nodes = [self._node_dict[arg_name] for arg_name in arg_names]
                    input_types: List[WeldSeriesPandas] = [self._type_map[arg_name] for arg_name in arg_names]
                    output_type = self._type_map[output_var_name]
                    assert (isinstance(output_type, WeldSeriesPandas))
                    series_concat_node = PandasSeriesConcatenationNode(input_nodes=input_nodes, input_names=arg_names,
                                                                       output_name=output_var_name,
                                                                       output_type=output_type, input_types=input_types)
                    return output_var_name, series_concat_node
                else:
                    return create_single_output_py_node(node)
            elif ".reshape" in called_function:
                if isinstance(value.func.value, ast.Name):
                    input_name = self.get_load_name(value.func.value.id, value.lineno, self._type_map)
                    input_node = self._node_dict[input_name]
                    output_type = self._type_map[output_var_name]
                    reshape_node = ReshapeNode(input_node=input_node, input_name=input_name,
                                               output_name=output_var_name, output_type=output_type,
                                               reshape_args=value.args)
                    return output_var_name, reshape_node
                else:
                    return create_single_output_py_node(node)
            elif called_function in self._async_funcs:
                return create_single_output_py_node(node, is_async_func=True)
            elif called_function in self._cached_funcs:
                return create_single_output_py_node(node, is_cached_func=True)
            else:
                return create_single_output_py_node(node)
        elif isinstance(value, ast.Attribute):
            if value.attr == "T" and isinstance(value.value, ast.Call) and isinstance(value.value.func, ast.Attribute) \
                    and isinstance(value.value.func.value, ast.Name) and value.value.func.attr == "to_frame":
                input_name = self.get_load_name(value.value.func.value.id, value.lineno, self._type_map)
                input_node = self._node_dict[input_name]
                input_type = self._type_map[input_name]
                assert (isinstance(input_type, WeldSeriesPandas))
                series_to_df_node = PandasSeriesToDataFrameNode(input_node=input_node, input_name=input_name,
                                                                input_type=input_type, output_name=output_var_name)
                return output_var_name, series_to_df_node
            elif value.attr == "values" and isinstance(value.value, ast.Name):
                input_name = self.get_load_name(value.value.id, value.lineno, self._type_map)
                if isinstance(self._type_map[input_name], WeldSeriesPandas):
                    input_node = self._node_dict[input_name]
                    output_type = self._type_map[output_var_name]
                    identity_node = IdentityNode(input_name=input_name, input_node=input_node,
                                                 output_name=output_var_name, output_type=output_type)
                    return output_var_name, identity_node
                else:
                    return create_single_output_py_node(node)
            else:
                return create_single_output_py_node(node)
        else:
            return create_single_output_py_node(node)

    @staticmethod
    def get_store_name(string, lineno):
        return "%s_%d" % (string, lineno)

    @staticmethod
    def get_load_name(string, lineno, type_map):
        most_recent_lineno: Optional[int] = None
        for i in range(lineno):
            if "%s_%d" % (string, i) in type_map:
                most_recent_lineno = i
        if most_recent_lineno is None:
            raise UntypedVariableException("Variable %s missing from type map" % string)
        else:
            return "%s_%d" % (string, most_recent_lineno)

    def analyze_Return(self, node: ast.Return) -> None:
        """
        Process the function return and create a graph which outputs whatever the function
        is returning.
        """
        return_names, return_py_node = self._create_py_node(node)
        output_node = WillumpOutputNode(return_py_node, return_names)
        self.willump_graph = WillumpGraph(output_node)

    def _create_py_node(self, entry: ast.AST, is_async_func=False, is_cached_node=False, does_not_modify_data=False) \
            -> Tuple[List[str], WillumpGraphNode]:
        entry_analyzer = ExpressionVariableAnalyzer(self._type_map)
        entry_analyzer.visit(entry)
        input_list, output_list = entry_analyzer.get_in_out_list()
        input_node_list = []
        for input_var in input_list:
            if input_var in self._node_dict:
                input_node_list.append(self._node_dict[input_var])
        output_types = [self._type_map[output_name] if output_name in self._type_map else None
                        for output_name in output_list]
        willump_python_node: WillumpPythonNode = WillumpPythonNode(python_ast=entry,
                                                                   input_names=input_list,
                                                                   output_names=output_list,
                                                                   output_types=output_types,
                                                                   in_nodes=input_node_list,
                                                                   is_async_node=is_async_func,
                                                                   is_cached_node=is_cached_node,
                                                                   does_not_modify_data=does_not_modify_data)
        return output_list, willump_python_node

    def _create_temp_var_from_node(self, entry: ast.expr, entry_type: WeldType) -> Tuple[str, WillumpGraphNode]:
        """
        Creates a variable equal to the value returned when evaluating an expression.  Returns the name of the variable
        and the node that generates it.
        """
        temp_var_name_node: ast.Name = ast.Name()
        temp_var_name_node.id = "__TEMP__%d" % self._temp_var_counter
        self._type_map[self.get_store_name(temp_var_name_node.id, entry.lineno)] = entry_type
        self._temp_var_counter += 1
        temp_var_name_node.ctx = ast.Store()
        temp_var_name_node.lineno = entry.lineno
        new_assign_node: ast.Assign = ast.Assign()
        new_assign_node.targets = [temp_var_name_node]
        new_assign_node.value = entry
        new_assign_node.lineno = entry.lineno
        return self.analyze_Assign(new_assign_node)

    def get_willump_graph(self) -> WillumpGraph:
        return self.willump_graph

    def get_args_list(self) -> List[str]:
        return self.arg_list

    def get_aux_data(self) -> List[Tuple[int, WeldType]]:
        return self.aux_data

    @staticmethod
    def get_assignment_target_name(node: ast.expr) -> Optional[str]:
        """
        Return the name of the target of an assignment statement.
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return WillumpGraphBuilder.get_assignment_target_name(node.value)
        else:
            return None

    @staticmethod
    def _get_function_name(call: ast.Call) -> Optional[str]:
        """
        Get the name of a function being called from a Call node.

        TODO:  Handle the ways different import statements can affect names. (e.g. numpy vs np).
        """

        def _get_layer_name(func) -> Optional[str]:
            if isinstance(func, ast.Name):
                return func.id
            elif isinstance(func, ast.Attribute):
                next_name = _get_layer_name(func.value)
                if next_name is None:
                    return None
                else:
                    return next_name + "." + func.attr
            elif isinstance(func, ast.Subscript):
                return _get_layer_name(func.value)
            else:
                return None

        return _get_layer_name(call.func)


class ExpressionVariableAnalyzer(ast.NodeVisitor):
    def __init__(self, type_map) -> None:
        self._output_list: List[str] = []
        self._input_list: List[str] = []
        self._type_map = type_map

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Store):
            self._output_list.append(WillumpGraphBuilder.get_store_name(node.value.id, node.lineno))
        else:
            try:
                self._input_list.append(WillumpGraphBuilder.get_load_name(node.value.id, node.lineno, self._type_map))
            except UntypedVariableException:
                pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self._output_list.append(WillumpGraphBuilder.get_store_name(node.id, node.lineno))
        else:
            try:
                self._input_list.append(WillumpGraphBuilder.get_load_name(node.id, node.lineno, self._type_map))
            except UntypedVariableException:
                pass

    def get_in_out_list(self) -> Tuple[List[str], List[str]]:
        return self._input_list, self._output_list


class UntypedVariableException(Exception):
    pass
