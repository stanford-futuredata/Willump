import typing
import willump.graph.willump_graph_node as wgn
from willump import panic


class MathOperation(object):
    """
    A representation of a math operation.  A TransformMathNode executes a sequence of these.
    """
    # Type of operation being performed.
    op_type: str
    # Is operation binary or unary?
    op_is_binop: bool
    # Should the value be appended to the transform node's output vector?
    append_to_output: bool
    # If first input is a variable, its name.
    first_input_var: typing.Optional[str]
    # If first input is a literal, its value.
    first_input_literal: typing.Optional[float]
    # If first input is from input array, the index.
    first_input_index: typing.Optional[int]
    # If second input is a variable, its name (unneeded if unary op).
    second_input_var: typing.Optional[str]
    # If second input is a literal, its value (unneeded if unary op).
    second_input_literal: typing.Optional[float]
    # If second input is from input array, the index.
    second_input_index: typing.Optional[int]
    # If the result should be stored for later, under what name?
    store_result_var_name: typing.Optional[str]

    def __init__(self, op_type: str, append_to_output: bool, first_input_var: str = None,
                 first_input_index: int = None,
                 first_input_literal: float = None, second_input_var: str = None,
                 second_input_index: int = None,
                 second_input_literal: float = None, store_result_var_name: str = None) -> None:
        self.op_type = op_type
        if op_type == "+" or op_type == "-" or op_type == "*" or op_type == "/":
            self.op_is_binop = True
        elif op_type == "sqrt":
            self.op_is_binop = False
        else:
            panic("Op not recognized")
        self.append_to_output = append_to_output
        self.first_input_var = first_input_var
        self.first_input_literal = first_input_literal
        self.first_input_index = first_input_index
        self.second_input_var = second_input_var
        self.second_input_literal = second_input_literal
        self.second_input_index = second_input_index
        self.store_result_var_name = store_result_var_name


class TransformMathNode(wgn.WillumpGraphNode):
    """
    Willump math transform node.  Takes in a series of MathOperations and executes them,
    returning a vector of the results of all transformations that specified appending to output.
    """
    input_nodes: typing.List[wgn.WillumpGraphNode]
    input_mathops: typing.List[MathOperation]
    _temp_var_counter: int = 0

    def __init__(self, input_node: wgn.WillumpGraphNode,
                 input_mathops: typing.List[MathOperation]) -> None:
        self.input_nodes = [input_node]
        self.input_mathops = input_mathops

    def get_in_nodes(self) -> typing.List[wgn.WillumpGraphNode]:
        return self.input_nodes

    def get_node_type(self) -> str:
        return "math"

    def _get_tmp_var_name(self):
        _temp_var_name = "__temp" + str(self._temp_var_counter)
        self._temp_var_counter += 1
        return _temp_var_name

    def get_node_weld(self) -> str:
        weld_str: str = "let __ret_array0 = appender[f64];"
        ret_array_counter: int = 0
        for i, mathop in enumerate(self.input_mathops):
            if mathop.store_result_var_name is not None:
                var_name: str = mathop.store_result_var_name
            else:
                var_name = self._get_tmp_var_name()
            # Define the first argument to the operation.
            first_arg: str
            if mathop.first_input_literal is not None:
                first_arg = str(mathop.first_input_literal)
            elif mathop.first_input_var is not None:
                first_arg = mathop.first_input_var
            elif mathop.first_input_index is not None:
                first_arg = "lookup({0}, {1}L)".format("{0}", mathop.first_input_index)
            else:
                panic("Math node without first argument.")
            # Process unary operations.
            if not mathop.op_is_binop:
                # Build the unary operation.
                weld_str += "let {0} = {1}({2});".format(var_name, mathop.op_type, first_arg)
            # Process binary operations.
            else:
                # Binary operations have a second argument, define it.
                second_arg: str
                if mathop.second_input_literal is not None:
                    second_arg = str(mathop.second_input_literal)
                elif mathop.second_input_var is not None:
                    second_arg = mathop.second_input_var
                elif mathop.second_input_index is not None:
                    second_arg = "lookup({0}, {1}L)".format("{0}", mathop.second_input_index)
                else:
                    panic("Math node binop without second argument.")
                # Build the binary operation.
                weld_str += "let {0} = {1} {2} {3};".format(var_name, first_arg,
                                                            mathop.op_type, second_arg)
            # Append the output of the operation to the output array if desired.
            if mathop.append_to_output:
                weld_str += "let __ret_array{0} = merge(__ret_array{1}, {2});" \
                    .format(ret_array_counter + 1, ret_array_counter, var_name)
                ret_array_counter += 1
        weld_str += "result(__ret_array{0})".format(ret_array_counter)
        return weld_str
