from typing import Optional, List, Tuple

from willump.graph.willump_graph_node import WillumpGraphNode
from willump import panic
from weld.types import *
import willump.willump_utilities as wutil


class MathOperationInput(object):
    input_type: str
    input_var: Optional[str]
    input_index: Optional[Tuple[str, int]]
    input_literal: Optional[float]

    def __init__(self, input_type: WeldType, input_var: str = None,
                 input_index: Tuple[str, int] = None, input_literal: float = None):
        self.input_type = wutil.weld_scalar_type_to_str(input_type)
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
    # Weld type of output of operation.
    output_type: str
    # What type (if any) the result of the operation must be casted to (either "" or output_type)
    output_cast: str = ""
    # What type (if any) the first input needs be cast to.
    first_input_cast: str = ""
    # If first input is a variable, its name.
    first_input_var: Optional[str] = None
    # If first input is a literal, its value.
    first_input_literal: Optional[float] = None
    # If first input is from input array, the index.
    first_input_index: Optional[Tuple[str, int]] = None
    # What type (if any) the second input needs be cast to (unneeded if unary op).
    second_input_cast: Optional[str] = None
    # If second input is a variable, its name (unneeded if unary op).
    second_input_var: Optional[str] = None
    # If second input is a literal, its value (unneeded if unary op).
    second_input_literal: Optional[float] = None
    # If second input is from input array, the index.
    second_input_index: Optional[Tuple[str, int]] = None

    @staticmethod
    def _casting_determination(first_input_type: str,
                               second_input_type: str, output_type: str) -> Tuple[str, str, str]:
        """
        Determine type casting in a binary operation.  If the two inputs have different types,
        cast the less precise type to the more precise one.  If the output has a different
        type than the more precise input, cast the output to that type.  This is meant to force
        Willump/Weld to comply with Python/Numpy semantics for mixed scalar types.
        """
        type_precision = {"i8": 0, "i16": 1, "i32": 2, "i64": 3, "f32": 4, "f64": 5}
        first_input_cast = second_input_cast = output_cast = ""
        if type_precision[first_input_type] < type_precision[second_input_type]:
            first_input_cast = second_input_type
        elif type_precision[first_input_type] > type_precision[second_input_type]:
            second_input_cast = first_input_type
        else:  # Equality
            pass
        if type_precision[output_type] !=\
                max(type_precision[first_input_type], type_precision[second_input_type]):
            output_cast = output_type
        return first_input_cast, second_input_cast, output_cast

    def __init__(self, op_type: str, output_var_name: str,
                 output_type: WeldType,
                 first_input: MathOperationInput,
                 second_input: MathOperationInput = None) -> None:
        self.op_type = op_type
        if op_type == "+" or op_type == "-" or op_type == "*" or op_type == "/":
            self.op_is_binop = True
        elif op_type == "sqrt" or op_type == "log":
            self.op_is_binop = False
        else:
            panic("Op not recognized")
        self.output_var_name = output_var_name
        self.output_is_vector = isinstance(output_type, WeldVec)
        self.output_type = wutil.weld_scalar_type_to_str(output_type)
        self.first_input_var = first_input.input_var
        self.first_input_literal = first_input.input_literal
        self.first_input_index = first_input.input_index
        if second_input is not None:
            self.second_input_var = second_input.input_var
            self.second_input_literal = second_input.input_literal
            self.second_input_index = second_input.input_index
            self.first_input_cast, self.second_input_cast, self.output_cast = \
                self._casting_determination(first_input.input_type,
                second_input.input_type, self.output_type)
        else:
            if op_type == "sqrt" or op_type == "log"\
                    and not wutil.weld_scalar_type_fp(weld_type_str=first_input.input_type):
                # Weld requires these operations have float inputs.
                self.first_input_cast = "f64"
            if self.output_type != first_input.input_type:
                self.output_cast = self.output_type

    def __repr__(self)-> str:
        return "Op: {0} Output: {1} {2} First:({3} {4} {5}) Second:({6} {7} {8})".format(
                self.op_type,
                self.output_var_name, self.output_is_vector, self.first_input_var,
                self.first_input_literal, self.first_input_index, self.second_input_var,
                self.second_input_literal, self.second_input_index)


class TransformMathNode(WillumpGraphNode):
    """
    Willump math transform node.  Takes in a series of MathOperations and executes them,
    returning a vector of the results of all transformations that specified appending to output.
    """
    input_nodes: List[WillumpGraphNode]
    input_mathops: List[MathOperation]
    output_name: str
    output_type: WeldType
    _temp_var_counter: int = 0

    def __init__(self, input_nodes: List[WillumpGraphNode],
                 input_mathops: List[MathOperation],
                 output_name: str,
                 output_type: WeldType) -> None:
        self.input_nodes = input_nodes
        self.input_mathops = input_mathops
        self.output_name = output_name
        self.output_type = output_type

    def get_in_nodes(self) -> List[WillumpGraphNode]:
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
        weld_str: str = "let __ret_array0 = appender[{0}];"\
            .format(wutil.weld_scalar_type_to_str(self.output_type))
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
                weld_str += "let {0} = {4}({1}({3}({2})));".format(var_name, mathop.op_type,
                            first_arg, mathop.first_input_cast, mathop.output_cast)
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
                weld_str += "let {0} = {6}({4}({1}) {2} {5}({3}));".format(var_name,
                                                                           first_arg, mathop.op_type, second_arg, mathop.first_input_cast,
                                                                           mathop.second_input_cast, mathop.output_cast)
            # Append the output of the operation to the output array if desired.
            if mathop.output_is_vector:
                weld_str += "let __ret_array{0} = merge(__ret_array{1}, {2});" \
                    .format(ret_array_counter + 1, ret_array_counter, var_name)
                ret_array_counter += 1
        weld_str += "let {0} = result(__ret_array{1});".format(self.output_name, ret_array_counter)
        return weld_str

    def __repr__(self) -> str:
        return "Transform math node with output {0}\n Inputs: {1}\n Transform: {2}\n"\
            .format(self.output_name,
                    str(list(map(lambda node: node.get_output_name(), self.input_nodes))),
                    self.input_mathops.__repr__())

