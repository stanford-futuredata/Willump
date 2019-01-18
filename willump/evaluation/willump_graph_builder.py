import ast

from willump import *
from willump.willump_utilities import *
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.string_lower_node import StringLowerNode
from willump.graph.string_removechar_node import StringRemoveCharNode
from willump.graph.array_count_vectorizer_node import ArrayCountVectorizerNode
from willump.graph.linear_regression_node import LinearRegressionNode
from willump.graph.array_append_node import ArrayAppendNode
from willump.graph.array_binop_node import ArrayBinopNode
from willump.graph.willump_hash_join_node import WillumpHashJoinNode
from willump.graph.willump_python_node import WillumpPythonNode

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

    def __init__(self, type_map: MutableMapping[str, WeldType],
                 static_vars: Mapping[str, object], batch=True) -> None:
        self._node_dict = {}
        self._type_map = type_map
        self._static_vars = static_vars
        self.arg_list = []
        self.aux_data = []
        self._temp_var_counter = 0
        self.batch = batch

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
                if assignment_node is None:
                    pass
                else:
                    self._node_dict[output_var_name] = assignment_node
            elif isinstance(entry, ast.Return):
                self.analyze_Return(entry)
            elif isinstance(entry, ast.For):
                for for_entry in entry.body:
                    output_var_name, assignment_node = self.analyze_Assign(for_entry)
                    if assignment_node is None:
                        pass
                    else:
                        self._node_dict[output_var_name] = assignment_node
            else:
                panic("Unrecognized body node %s" % ast.dump(entry))

    def analyze_Assign(self, node: ast.Assign) -> \
            Tuple[str, Optional[WillumpGraphNode]]:
        """
        Process an assignment AST node into either a Willump Weld node or Willump Python node that
        defines the variable being assigned.
        """
        assert (len(node.targets) == 1)  # Assume assignment to only one variable.
        target: ast.expr = node.targets[0]
        output_var_name = self.get_assignment_target_name(target)
        if output_var_name is None:
            return self._create_py_node(node)
        output_type = self._type_map[output_var_name]
        value: ast.expr = node.value
        if isinstance(value, ast.BinOp):
            if not isinstance(output_type, WeldVec):
                return self._create_py_node(node)
            if isinstance(target, ast.Subscript):
                return self._create_py_node(node)
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
                array_binop_node = ArrayBinopNode(left_node, right_node, output_var_name, output_type, "+")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Sub):
                array_binop_node = ArrayBinopNode(left_node, right_node, output_var_name, output_type, "-")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Mult):
                array_binop_node = ArrayBinopNode(left_node, right_node, output_var_name, output_type, "*")
                return output_var_name, array_binop_node
            elif isinstance(value.op, ast.Div):
                array_binop_node = ArrayBinopNode(left_node, right_node, output_var_name, output_type, "/")
                return output_var_name, array_binop_node
            else:
                return self._create_py_node(node)
        elif isinstance(value, ast.Call):
            called_function: Optional[str] = self._get_function_name(value)
            if called_function is None:
                return self._create_py_node(node)
            # TODO:  Update this for batched code.
            if "split" in called_function:
                split_input_var: str = value.func.value.id
                split_input_node: WillumpGraphNode = self._node_dict[split_input_var]
                string_split_node: StringSplitNode = StringSplitNode(input_node=split_input_node,
                                                                     output_name=output_var_name)
                return output_var_name, string_split_node
            # TODO:  Update this for batched code.
            elif "lower" in called_function:
                lower_input_var: str = value.func.value.value.id
                lower_input_node: WillumpGraphNode = self._node_dict[lower_input_var]
                string_lower_node: StringLowerNode = StringLowerNode(input_node=lower_input_node,
                                                                     output_name=output_var_name)
                return output_var_name, string_lower_node
            # TODO:  Update this for batched code.
            elif "replace" in called_function:
                replace_input_var: str = value.func.value.value.id
                replace_input_node: WillumpGraphNode = self._node_dict[replace_input_var]
                target_char: str = value.args[0].s
                assert (len(target_char) == 1)
                assert (len(value.args[1].s) == 0)
                string_remove_char_node: StringRemoveCharNode = \
                    StringRemoveCharNode(input_node=replace_input_node,
                                         target_char=target_char, output_name=output_var_name)
                return output_var_name, string_remove_char_node
            # TODO:  Lots of potential predictors, differentiate them!
            elif "predict" in called_function:
                logit_input_var: str = value.args[0].id
                logit_weights = self._static_vars[WILLUMP_LINEAR_REGRESSION_WEIGHTS]
                logit_intercept = self._static_vars[WILLUMP_LINEAR_REGRESSION_INTERCEPT]
                logit_input_node: WillumpGraphNode = self._node_dict[logit_input_var]
                logit_node: LinearRegressionNode = LinearRegressionNode(
                    input_node=logit_input_node, input_type=self._type_map[logit_input_var],
                    output_name=output_var_name,
                    output_type = self._type_map[output_var_name],
                    logit_weights=logit_weights,
                    logit_intercept=logit_intercept, aux_data=self.aux_data, batch=self.batch
                )
                return output_var_name, logit_node
            # TODO:  Update this for batched code.
            elif "numpy.append" in called_function:
                append_input_array: str = value.args[0].id
                append_input_value: str = value.args[1].id
                append_input_array_node: WillumpGraphNode = self._node_dict[append_input_array]
                append_input_val_node: WillumpGraphNode = self._node_dict[append_input_value]
                array_append_node: ArrayAppendNode = ArrayAppendNode(append_input_array_node, append_input_val_node,
                                                                     output_var_name,
                                                                     self._type_map[append_input_array],
                                                                     self._type_map[output_var_name])
                return output_var_name, array_append_node
            # TODO:  Don't assume name is input_vect.
            elif "input_vect.transform" in called_function:
                if self._static_vars[WILLUMP_COUNT_VECTORIZER_LOWERCASE] is True \
                        or self._static_vars[WILLUMP_COUNT_VECTORIZER_ANALYZER] is not 'char':
                    return self._create_py_node(node)
                freq_count_input_var: str = value.args[0].id
                vocab_dict = self._static_vars[WILLUMP_COUNT_VECTORIZER_VOCAB]
                array_cv_input_node: WillumpGraphNode = self._node_dict[freq_count_input_var]
                array_cv_node: ArrayCountVectorizerNode = ArrayCountVectorizerNode(
                    input_node=array_cv_input_node, output_name=output_var_name,
                    input_vocab_dict=vocab_dict, aux_data=self.aux_data,
                    ngram_range=self._static_vars[WILLUMP_COUNT_VECTORIZER_NGRAM_RANGE]
                )
                return output_var_name, array_cv_node
            elif ".merge" in called_function:
                if self._static_vars[WILLUMP_JOIN_HOW + str(node.lineno)] is not 'left':
                    return self._create_py_node(node)
                join_col: str = self._static_vars[WILLUMP_JOIN_COL + str(node.lineno)]
                left_df_columns: List[str] = list(self._static_vars[WILLUMP_JOIN_LEFT_COLUMNS + str(node.lineno)])
                right_df = self._static_vars[WILLUMP_JOIN_RIGHT_DATAFRAME + str(node.lineno)]
                left_df_input_var: str = value.func.value.id
                left_df_input_node = self._node_dict[left_df_input_var]
                willump_hash_join_node = WillumpHashJoinNode(input_node=left_df_input_node, output_name=output_var_name,
                                                             join_col_name=join_col,
                                                             right_dataframe=right_df, aux_data=self.aux_data,
                                                             batch=self.batch, size_left_df=len(left_df_columns),
                                                             join_col_left_index=left_df_columns.index(join_col))
                return output_var_name, willump_hash_join_node
            # TODO:  What to do with these?
            elif "reshape" in called_function:
                return output_var_name, None
            else:
                return self._create_py_node(node)
        else:
            return self._create_py_node(node)

    def analyze_Return(self, node: ast.Return) -> None:
        """
        Process the function return and create a graph which outputs whatever the function
        is returning.

        Assumes function returns a single value, which must be a numpy float64 array.
        """
        output_node: WillumpOutputNode
        if isinstance(node.value, ast.Name):
            output_name: str = node.value.id
            if output_name in self._node_dict:
                output_node = WillumpOutputNode(WillumpPythonNode(node, output_name, [self._node_dict[output_name]]))
            else:
                panic("No in-node found for return node {0}".format(ast.dump(node)))
            self.willump_graph = WillumpGraph(output_node)
        else:
            panic("Unrecognized return: {0}".format(ast.dump(node)))

    def _create_py_node(self, entry: ast.AST) -> Tuple[str, WillumpGraphNode]:
        entry_analyzer = ExpressionVariableAnalyzer()
        entry_analyzer.visit(entry)
        input_list, output_list = entry_analyzer.get_in_out_list()
        assert (len(output_list) == 1)
        output_var_name = output_list[0]
        input_node_list = []
        for input_var in input_list:
            if input_var in self._node_dict:
                input_node_list.append(self._node_dict[input_var])
        willump_python_node: WillumpPythonNode = WillumpPythonNode(entry, output_var_name, input_node_list)
        return output_var_name, willump_python_node

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
