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
from willump.graph.willump_hash_join_node import WillumpHashJoinNode
from willump.graph.willump_python_node import WillumpPythonNode
from willump.graph.pandas_column_selection_node import PandasColumnSelectionNode
from willump.graph.stack_sparse_node import StackSparseNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode

from typing import MutableMapping, List, Tuple, Optional, Mapping
from weld.types import *


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from the Python AST for a FunctionDef.  Typically called from a
    decorator around that function.  Makes the following assumptions:

    1.  No variable ever changes its type.

    2.  No control flow changes.

    # TODO:  Allow control flow changes.
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
                 static_vars: Mapping[str, object], async_funcs: List[str], batch: bool) -> None:
        self._node_dict = {}
        self._type_map = type_map
        self._static_vars = static_vars
        self.arg_list = []
        self.aux_data = []
        self._temp_var_counter = 0
        self.batch = batch
        self._async_funcs = async_funcs

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Begin processing of a function.  Create input nodes for function arguments.
        """
        for arg in node.args.args:
            arg_name: str = arg.arg
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

        def create_single_output_py_node(in_node, is_async_func=False):
            output_names, py_node = self._create_py_node(in_node, is_async_func=is_async_func)
            assert (len(output_names) == 1)
            return output_names[0], py_node

        assert (len(node.targets) == 1)  # Assume assignment to only one variable.
        target: ast.expr = node.targets[0]
        output_var_name = self.get_assignment_target_name(target)
        if output_var_name is None:
            return create_single_output_py_node(node)
        output_type = self._type_map[output_var_name]
        value: ast.expr = node.value
        if isinstance(value, ast.BinOp):
            if not isinstance(output_type, WeldVec):
                return create_single_output_py_node(node)
            if isinstance(target, ast.Subscript):
                return create_single_output_py_node(node)
            if isinstance(value.left, ast.Name) and value.left.id in self._node_dict:
                left_name: str = value.left.id
                left_node: WillumpGraphNode = self._node_dict[left_name]
            else:
                # TODO:  Type the new node properly.
                left_name, left_node = self._create_temp_var_from_node(value.left, output_type)
                self._node_dict[left_name] = left_node
            if isinstance(value.right, ast.Name) and value.right.id in self._node_dict:
                right_name: str = value.right.id
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
            if isinstance(value.slice.value, ast.Name) and isinstance(value.value, ast.Name):
                input_var_name = value.value.id
                column_names = self._static_vars[WILLUMP_SUBSCRIPT_INDEX_NAME + str(value.lineno)]
                if isinstance(self._type_map[input_var_name], WeldPandas) and isinstance(column_names, list):
                    pandas_column_selection_node = \
                        PandasColumnSelectionNode(input_node=self._node_dict[input_var_name],
                                                  input_name=input_var_name,
                                                  output_name=output_var_name,
                                                  input_type=self._type_map[input_var_name],
                                                  selected_columns=column_names)
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
                    input_df_name = value.func.value.value.id
                    input_df_node = self._node_dict[input_df_name]
                    input_df_type = self._type_map[input_df_name]

                    target_col = value.func.value.slice.value.s
                    if isinstance(value.args[0], ast.Attribute) and value.args[0].attr == "lower":
                        string_lower_node: StringLowerNode = StringLowerNode(input_node=input_df_node,
                                                                             input_name=input_df_name,
                                                                             input_type=input_df_type,
                                                                             output_name=output_var_name,
                                                                             target_col=target_col)
                        return output_var_name, string_lower_node
                    else:
                        return create_single_output_py_node(node)
                else:
                    return create_single_output_py_node(node)
            # TODO:  Lots of potential predictors, differentiate them!
            elif ".predict" in called_function:
                if WILLUMP_LINEAR_REGRESSION_WEIGHTS in self._static_vars:
                    logit_input_var: str = value.args[0].id
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
                        logit_intercept=logit_intercept, aux_data=self.aux_data, batch=self.batch
                    )
                    return output_var_name, logit_node
                else:
                    return create_single_output_py_node(node)
            elif ".transform" in called_function:
                lineno = str(node.lineno)
                if WILLUMP_COUNT_VECTORIZER_VOCAB + lineno not in self._static_vars or \
                        self._static_vars[WILLUMP_COUNT_VECTORIZER_LOWERCASE + lineno] is True:
                    return create_single_output_py_node(node)
                vectorizer_input_var: str = value.args[0].id
                analyzer: str = self._static_vars[WILLUMP_COUNT_VECTORIZER_ANALYZER + lineno]
                vocab_dict: Mapping[str, int] = self._static_vars[WILLUMP_COUNT_VECTORIZER_VOCAB + lineno]
                vectorizer_input_node: WillumpGraphNode = self._node_dict[vectorizer_input_var]
                vectorizer_ngram_range: Tuple[int, int] = \
                    self._static_vars[WILLUMP_COUNT_VECTORIZER_NGRAM_RANGE + lineno]
                # TODO:  Once the Weld compilation segfault is fixed, replace with Weld node.
                array_space_combiner_output = vectorizer_input_var + "__weld_combined__"
                if array_space_combiner_output in self._node_dict:
                    array_space_combiner_node = self._node_dict[array_space_combiner_output]
                else:
                    array_space_combiner_python = "%s = list(map(lambda x: re.sub(r'\s+', ' ', x), %s))" \
                                                  % (array_space_combiner_output, vectorizer_input_var)
                    array_space_combiner_ast: ast.Module = ast.parse(array_space_combiner_python)
                    array_space_combiner_node: WillumpPythonNode = WillumpPythonNode(
                        python_ast=array_space_combiner_ast.body[0],
                        input_names=[vectorizer_input_var],
                        output_names=[array_space_combiner_output],
                        in_nodes=[vectorizer_input_node])
                    self._type_map[array_space_combiner_output] = self._type_map[vectorizer_input_var]
                    self._node_dict[array_space_combiner_output] = array_space_combiner_node
                if WILLUMP_TFIDF_IDF_VECTOR + lineno in self._static_vars:
                    idf_vector = self._static_vars[WILLUMP_TFIDF_IDF_VECTOR + lineno]
                    tfidf_node: ArrayTfIdfNode = ArrayTfIdfNode(
                        input_node=array_space_combiner_node,
                        input_name=array_space_combiner_output,
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
                        input_node=array_space_combiner_node,
                        input_name=array_space_combiner_output,
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
                left_df_input_var: str = value.func.value.id
                left_df_input_node = self._node_dict[left_df_input_var]
                willump_hash_join_node = WillumpHashJoinNode(input_node=left_df_input_node,
                                                             input_name=left_df_input_var,
                                                             output_name=output_var_name,
                                                             left_input_type=self._type_map[left_df_input_var],
                                                             join_col_name=join_col,
                                                             right_dataframe=right_df, aux_data=self.aux_data,
                                                             batch=self.batch)
                return output_var_name, willump_hash_join_node
            elif "sparse.hstack" in called_function:
                if not isinstance(value.args[0], ast.List) or \
                        not all(isinstance(ele, ast.Name) for ele in value.args[0].elts):
                    return create_single_output_py_node(node)
                input_nodes = [self._node_dict[arg_node.id] for arg_node in value.args[0].elts]
                input_names = [arg_node.id for arg_node in value.args[0].elts]
                output_type = self._type_map[output_var_name]
                stack_sparse_node = StackSparseNode(input_nodes=input_nodes,
                                                    input_names=input_names,
                                                    output_name=output_var_name,
                                                    output_type=output_type)
                return output_var_name, stack_sparse_node
            elif called_function in self._async_funcs:
                return create_single_output_py_node(node, is_async_func=True)
            else:
                return create_single_output_py_node(node)
        else:
            return create_single_output_py_node(node)

    def analyze_Return(self, node: ast.Return) -> None:
        """
        Process the function return and create a graph which outputs whatever the function
        is returning.

        Assumes function returns a single value, which must be a numpy float64 array.
        """
        return_names, return_py_node = self._create_py_node(node)
        output_node = WillumpOutputNode(return_py_node, return_names)
        self.willump_graph = WillumpGraph(output_node)

    def _create_py_node(self, entry: ast.AST, is_async_func=False) -> Tuple[List[str], WillumpGraphNode]:
        entry_analyzer = ExpressionVariableAnalyzer()
        entry_analyzer.visit(entry)
        input_list, output_list = entry_analyzer.get_in_out_list()
        input_node_list = []
        for input_var in input_list:
            if input_var in self._node_dict:
                input_node_list.append(self._node_dict[input_var])
        willump_python_node: WillumpPythonNode = WillumpPythonNode(python_ast=entry,
                                                                   input_names=input_list,
                                                                   output_names=output_list,
                                                                   in_nodes=input_node_list,
                                                                   is_async_node=is_async_func)
        return output_list, willump_python_node

    def _create_temp_var_from_node(self, entry: ast.expr, entry_type: WeldType) -> Tuple[str, WillumpGraphNode]:
        """
        Creates a variable equal to the value returned when evaluating an expression.  Returns the name of the variable
        and the node that generates it.
        """
        temp_var_name_node: ast.Name = ast.Name()
        temp_var_name_node.id = "__TEMP__%d" % self._temp_var_counter
        self._type_map[temp_var_name_node.id] = entry_type
        self._temp_var_counter += 1
        temp_var_name_node.ctx = ast.Store()
        new_assign_node: ast.Assign = ast.Assign()
        new_assign_node.targets = [temp_var_name_node]
        new_assign_node.value = entry
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
    def __init__(self) -> None:
        self._output_list: List[str] = []
        self._input_list: List[str] = []

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.ctx, ast.Store):
            self._output_list.append(node.value.id)
        else:
            self._input_list.append(node.value.id)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            self._output_list.append(node.id)
        else:
            self._input_list.append(node.id)

    def get_in_out_list(self) -> Tuple[List[str], List[str]]:
        return self._input_list, self._output_list
