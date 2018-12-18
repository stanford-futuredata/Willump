import ast

from willump import panic
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.string_split_node import StringSplitNode
from willump.graph.string_lower_node import StringLowerNode
from willump.graph.string_removechar_node import StringRemoveCharNode
from willump.graph.vocabulary_frequency_count_node import VocabularyFrequencyCountNode
from willump.graph.logistic_regression_node import LogisticRegressionNode
from willump.graph.array_append_node import ArrayAppendNode
from willump.graph.array_addition_node import ArrayAdditionNode
from willump.graph.willump_python_node import WillumpPythonNode

from typing import MutableMapping, List, Tuple, Optional, Mapping
from weld.types import *


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from the Python AST for a FunctionDef.  Typically called from a
    decorator around that function.  Makes the following assumptions:

    1.  Input Python is the definition of a single function,
        from which the graph shall be extracted.

    2.  The function does not reference anything outside of its own scope.

    TODO:  Implement UDFs to weaken assumption 2 as well as deal with syntax we don't recognize.
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

    def __init__(self, type_map: MutableMapping[str, WeldType],
                 static_vars: Mapping[str, object]) -> None:
        self._node_dict = {}
        self._type_map = type_map
        self._static_vars = static_vars
        self.arg_list = []
        self.aux_data = []

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
                result = self.analyze_Assign(entry)
                if result is None:
                    self._create_py_node(entry)
                else:
                    output_var_name, assignment_node = result
                    if assignment_node is None:
                        pass
                    else:
                        self._node_dict[output_var_name] = assignment_node
            elif isinstance(entry, ast.Return):
                self.analyze_Return(entry)
            elif isinstance(entry, ast.For):
                for for_entry in entry.body:
                    if isinstance(for_entry, ast.Assign):
                        result = self.analyze_Assign(for_entry)
                        if result is None:
                            panic("Unrecognized assign %s" % ast.dump(for_entry))
                        else:
                            output_var_name, assignment_node = result
                            if assignment_node is None:
                                pass
                            else:
                                self._node_dict[output_var_name] = assignment_node
            else:
                panic("Unrecognized body node %s" % ast.dump(entry))

    def analyze_Assign(self, node: ast.Assign) -> \
            Optional[Tuple[str, Optional[WillumpGraphNode]]]:
        """
        Process an assignment AST node into either a Willump node or MathOperation that
        defines the variable being assigned.
        """
        assert (len(node.targets) == 1)  # Assume assignment to only one variable.
        target: ast.expr = node.targets[0]
        output_var_name = self.get_assignment_target_name(target)
        if output_var_name is None:
            return None
        output_type = self._type_map[output_var_name]
        value: ast.expr = node.value
        if isinstance(value, ast.BinOp):
            if not isinstance(output_type, WeldVec):
                return None
            if isinstance(value.left, ast.Name):
                left_name: str = value.left.id
                left_node: WillumpGraphNode = self._node_dict[left_name]
            else:
                return None
            if isinstance(value.right, ast.Name):
                right_name: str = value.right.id
                right_node: WillumpGraphNode = self._node_dict[right_name]
            else:
                return None
            if isinstance(value.op, ast.Add):
                array_addition_node = ArrayAdditionNode(left_node, right_node, output_var_name, output_type)
                return output_var_name, array_addition_node
        elif isinstance(value, ast.Call):
            called_function: Optional[str] = self._get_function_name(value)
            if called_function is None:
                return None
            # TODO:  Recognize functions in attributes properly.
            if "split" in called_function:
                split_input_var: str = value.func.value.id
                split_input_node: WillumpGraphNode = self._node_dict[split_input_var]
                string_split_node: StringSplitNode = StringSplitNode(input_node=split_input_node,
                                                                     output_name=output_var_name)
                return output_var_name, string_split_node
            # TODO:  Recognize when this is being called in a loop, don't just assume it is.
            elif "lower" in called_function:
                lower_input_var: str = value.func.value.value.id
                lower_input_node: WillumpGraphNode = self._node_dict[lower_input_var]
                string_lower_node: StringLowerNode = StringLowerNode(input_node=lower_input_node,
                                                                     output_name=output_var_name)
                return output_var_name, string_lower_node
            # TODO:  Recognize replaces that do more than remove one character.
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
            # TODO:  Find a real function to use here.
            elif "willump_frequency_count" in called_function:
                freq_count_input_var: str = value.args[0].id
                vocab_dict = self._static_vars["willump_frequency_count_vocab"]
                freq_count_input_node: WillumpGraphNode = self._node_dict[freq_count_input_var]
                vocab_freq_count_node: VocabularyFrequencyCountNode = VocabularyFrequencyCountNode(
                    input_node=freq_count_input_node, output_name=output_var_name,
                    input_vocab_dict=vocab_dict, aux_data=self.aux_data
                )
                return output_var_name, vocab_freq_count_node
            # TODO:  Lots of potential predictors, differentiate them!
            elif "predict" in called_function:
                logit_input_var: str = value.args[0].id
                logit_weights = self._static_vars["willump_logistic_regression_weights"]
                logit_intercept = self._static_vars["willump_logistic_regression_intercept"]
                logit_input_node: WillumpGraphNode = self._node_dict[logit_input_var]
                logit_node: LogisticRegressionNode = LogisticRegressionNode(
                    input_node=logit_input_node, output_name=output_var_name,
                    logit_weights=logit_weights,
                    logit_intercept=logit_intercept, aux_data=self.aux_data
                )
                return output_var_name, logit_node
            # TODO:  Support values that are not scalar variables.
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
            # TODO:  What to do with these?
            elif "reshape" in called_function:
                return output_var_name, None
            else:
                return None
        else:
            return None

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

    def _create_py_node(self, entry: ast.AST) -> None:
        entry_analyzer = ExpressionVariableAnalyzer()
        entry_analyzer.visit(entry)
        input_list, output_list = entry_analyzer.get_in_out_list()
        # TODO:  Multiple outputs.
        assert (len(output_list) == 1)
        output_var_name = output_list[0]
        input_node_list = []
        for input_var in input_list:
            if input_var in self._node_dict:
                input_node_list.append(self._node_dict[input_var])
        willump_python_node: WillumpPythonNode = WillumpPythonNode(entry, output_var_name, input_node_list)
        self._node_dict[output_var_name] = willump_python_node
        return

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
