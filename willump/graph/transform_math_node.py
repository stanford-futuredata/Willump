from typing import Optional, List, Tuple

import willump.graph.willump_graph_node as wgn
from willump import panic


class MathOperationInput(object):
    input_var: Optional[str]
    input_index: Optional[Tuple[str, int]]
    input_literal: Optional[float]

    def __init__(self, input_var: str = None, input_index: Tuple[str, int] = None,
                 input_literal: float = None):
        self.input_var = input_var
        self.input_index = input_index
        self.input_literal = input_literal


class MathOperation(object):
    """
    A representation of a math operation.  A TransformMathNode executes a sequence of these.
    """
    # Type of operation being performed.
    op_type: str
    # Is operation binary or unary?
    op_is_binop: bool
    # What is the name of the variable the result goes into?
    output_var_name: str
    # Is the variable the result goes into a vector?
    output_is_vector: bool
    # If first input is a variable, its name.
    first_input_var: Optional[str]
    # If first input is a literal, its value.
    first_input_literal: Optional[float]
    # If first input is from input array, the index.
    first_input_index: Optional[Tuple[str, int]]
    # If second input is a variable, its name (unneeded if unary op).
    second_input_var: Optional[str]
    # If second input is a literal, its value (unneeded if unary op).
    second_input_literal: Optional[float]
    # If second input is from input array, the index.
    second_input_index: Optional[Tuple[str, int]]

    def __init__(self, op_type: str, output_var_name: str, output_is_vector: bool,
                 first_input: MathOperationInput,
                 second_input: MathOperationInput = None) -> None:
        self.op_type = op_type
        if op_type == "+" or op_type == "-" or op_type == "*" or op_type == "/":
            self.op_is_binop = True
        elif op_type == "sqrt":
            self.op_is_binop = False
        else:
            panic("Op not recognized")
        self.output_var_name = output_var_name
        self.output_is_vector = output_is_vector
        self.first_input_var = first_input.input_var
        self.first_input_literal = first_input.input_literal
        self.first_input_index = first_input.input_index
        if second_input is not None:
            self.second_input_var = second_input.input_var
            self.second_input_literal = second_input.input_literal
            self.second_input_index = second_input.input_index

    def __repr__(self)-> str:
        return "Op: {0} Output: {1} {2} First:({3} {4} {5}) Second:({6} {7} {8})".format(
                self.op_type,
                self.output_var_name, self.output_is_vector, self.first_input_var,
                self.first_input_literal, self.first_input_index, self.second_input_var,
                self.second_input_literal, self.second_input_index)


class TransformMathNode(wgn.WillumpGraphNode):
    """
    Willump math transform node.  Takes in a series of MathOperations and executes them,
    returning a vector of the results of all transformations that specified appending to output.
    """
    input_nodes: List[wgn.WillumpGraphNode]
    input_mathops: List[MathOperation]
    output_name: str
    _temp_var_counter: int = 0

    def __init__(self, input_node: wgn.WillumpGraphNode,
                 input_mathops: List[MathOperation],
                 output_name: str) -> None:
        self.input_nodes = [input_node]
        self.input_mathops = input_mathops
        self.output_name = output_name

    def get_in_nodes(self) -> List[wgn.WillumpGraphNode]:
        return self.input_nodes

    def get_node_type(self) -> str:
        return "math"

    def get_output_name(self) -> str:
        return self.output_name

    def _get_tmp_var_name(self):
        _temp_var_name = "__temp" + str(self._temp_var_counter)
        self._temp_var_counter += 1
        return _temp_var_name

    def get_node_weld(self) -> str:
        weld_str: str = "let __ret_array0 = appender[f64];"
        ret_array_counter: int = 0
        for i, mathop in enumerate(self.input_mathops):
            if not mathop.output_is_vector:
                var_name: str = mathop.output_var_name
            else:
                var_name = self._get_tmp_var_name()
            # Define the first argument to the operation.
            first_arg: str
            if mathop.first_input_literal is not None:
                first_arg = str(mathop.first_input_literal)
            elif mathop.first_input_var is not None:
                first_arg = mathop.first_input_var
            elif mathop.first_input_index is not None:
                first_arg = "lookup({0}, {1}L)".format(mathop.first_input_index[0],
                                                       mathop.first_input_index[1])
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
                    second_arg = "lookup({0}, {1}L)".format(mathop.second_input_index[0],
                                                            mathop.second_input_index[1])
                else:
                    panic("Math node binop without second argument.")
                # Build the binary operation.
                weld_str += "let {0} = {1} {2} {3};".format(var_name, first_arg,
                                                            mathop.op_type, second_arg)
            # Append the output of the operation to the output array if desired.
            if mathop.output_is_vector:
                weld_str += "let __ret_array{0} = merge(__ret_array{1}, {2});" \
                    .format(ret_array_counter + 1, ret_array_counter, var_name)
                ret_array_counter += 1
        weld_str += "let {0} = __ret_array{1};".format(self.output_name, ret_array_counter)
        return weld_str

    def __repr__(self) -> str:
        return "Transform math node\n Transform: {0}\n".format(self.input_mathops.__repr__())

