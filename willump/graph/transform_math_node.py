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
    # Should the value be appended to the transform node's output vector?
    append_to_output: bool

    def __init__(self, op_type: str, append_to_output: bool, first_input_var: str = None,
                 first_input_index: int = None,
                 first_input_literal: float = None, second_input_var: str = None,
                 second_input_index: int = None,
                 second_input_literal: float = None) -> None:
        self.op_type = op_type
        if op_type == "+":
            self.op_is_binop = True
        elif op_type == "sqrt":
            self.op_is_binop = False
        else:
            panic("Op not recognized")
        self.first_input_var = first_input_var
        self.first_input_literal = first_input_literal
        self.first_input_index = first_input_index
        self.second_input_var = second_input_var
        self.second_input_literal = second_input_literal
        self.second_input_index = second_input_index
        self.append_to_output = append_to_output


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
        for i, mathop in enumerate(self.input_mathops):
            tmp_var_name: str = self._get_tmp_var_name()
            first_arg: str
            if mathop.first_input_literal is not None:
                first_arg = str(mathop.first_input_literal)
            elif mathop.first_input_var is not None:
                panic("Unsupported")
            elif mathop.first_input_index is not None:
                first_arg = "lookup({0}, {1}L)".format("{0}", mathop.first_input_index)
            else:
                panic("Math node without first argument.")
            if mathop.op_is_binop:
                second_arg: str
                if mathop.second_input_literal is not None:
                    second_arg = str(mathop.second_input_literal)
                elif mathop.first_input_var is not None:
                    panic("Unsupported")
                elif mathop.first_input_index is not None:
                    second_arg = "lookup({0}, {1}L)".format("{0}", mathop.second_input_index)
                else:
                    panic("Math node binop without second argument.")
                if mathop.op_type == "+":
                    weld_str += "let {0} = {1} + {2};".format(tmp_var_name, first_arg, second_arg)
                else:
                    panic("Unsupported")
            else:
                panic("Unsupported")
            if mathop.append_to_output:
                weld_str += "let __ret_array{0} = merge(__ret_array{1}, {2});" \
                    .format(i+1, i, tmp_var_name)
        weld_str += "result(__ret_array{0})".format(len(self.input_mathops))
        return weld_str