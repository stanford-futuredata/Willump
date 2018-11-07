import ast

from willump import panic
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.transform_math_node import TransformMathNode, MathOperation, MathOperationInput

from typing import MutableMapping, List, Tuple, Optional, Set


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from raw Python.  Assumptions we currently make:

    1.  Input Python contains a single function, from which the graph shall be extracted.

    2.  This function takes in and returns a numpy.float64 array.

    3.  The function does not reference anything outside of its own scope.
    """
    willump_graph: WillumpGraph
    # A map from the names of variables to the nodes that generate them.
    _node_dict: MutableMapping[str, WillumpGraphNode] = {}
    _mathops_list: List[MathOperation] = []

    def __init__(self):
        self._node_dict = {}
        self._mathops_list = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Begin processing of a function.  Create input nodes for function arguments.

        Assumes all function arguments are numpy float64 arrays.
        """
        for arg in node.args.args:
            arg_name: str = arg.arg
            input_node: WillumpInputNode = WillumpInputNode(arg_name)
            self._node_dict[arg_name] = input_node
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node: ast.Assign):
        """
        Process an assignment AST node into either a Willump node or MathOperation that
        defines the variable being assigned.
        """
        # Assume assignment to only one variable.
        assert(len(node.targets) == 1)
        target: ast.expr = node.targets[0]
        output_var_name, output_is_vector = self._analyze_assignment_target(target)
        value: ast.expr = node.value
        if isinstance(value, ast.BinOp):
            left, op, right = self._analyze_binop(value)
            binop_mathop: MathOperation = MathOperation(op, output_var_name, output_is_vector,
                                                        left, second_input=right)
            self._mathops_list.append(binop_mathop)

    def visit_Return(self, node: ast.Return):
        """
        Process the function return and create a graph which outputs whatever the function
        is returning.

        Assumes function returns a single value, which must be a numpy float64 array.
        """
        output_node: WillumpOutputNode
        if isinstance(node.value, ast.Name):
            output_name: str = node.value.id
            if output_name in self._node_dict:
                output_node = WillumpOutputNode(self._node_dict[output_name])
            else:
                potential_in_node: Optional[TransformMathNode] = \
                    self._build_math_transform_for_output(output_name)
                if potential_in_node is not None:
                    output_node = WillumpOutputNode(potential_in_node)
                else:
                    panic("No in-node found for return node {0}".format(ast.dump(node)))
            self.willump_graph = WillumpGraph(output_node)
        else:
            panic("Unrecognized return: {0}".format(ast.dump(node)))

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def get_willump_graph(self) -> WillumpGraph:
        return self.willump_graph

    def _analyze_assignment_target(self, node: ast.expr) -> Tuple[str, bool]:
        """
        Return the name and type of the target of an assignment statement.
        """
        if isinstance(node, ast.Name):
            return node.id, False
        elif isinstance(node, ast.Subscript):
            node_name, _ = self._analyze_assignment_target(node.value)
            return node_name, True
        else:
            panic("Unrecognized assignment to: {0}".format(ast.dump(node)))
            return "", False

    def _expr_to_math_operation_input(self, expr: ast.expr) -> MathOperationInput:
        """
        Convert an expression input to a binary or unary operation into a MathOperationInput.
        """
        if isinstance(expr, ast.Num):
            return MathOperationInput(input_literal=expr.n)
        elif isinstance(expr, ast.Subscript):
            return MathOperationInput(input_index=(expr.value.id, expr.slice.value.n))
        elif isinstance(expr, ast.Name):
            return MathOperationInput(input_var=expr.id)
        else:
            panic("Unrecognized expression in binop {0}".format(ast.dump(expr)))
            return MathOperationInput()

    def _analyze_binop(self, binop: ast.BinOp) -> \
            Tuple[MathOperationInput, str, MathOperationInput]:
        """
        Convert an AST binop into two MathOperationInputs and an operator-string for a Transform
        Math Node.
        """
        operator: str = self._pybinop_to_wbinop(binop.op)
        if operator == "":
            panic("Unsupported binop {0}".format(ast.dump(binop)))
        left_expr: ast.expr = binop.left
        left_input: MathOperationInput = self._expr_to_math_operation_input(left_expr)
        right_expr: ast.expr = binop.right
        right_input: MathOperationInput = self._expr_to_math_operation_input(right_expr)
        return left_input, operator, right_input

    def _build_math_transform_for_output(self, output_name: str) -> Optional[TransformMathNode]:
        """
        Build a TransformMathNode that will output a particular variable.  Return None
        if no such node can be built.

        TODO:  Don't recalculate intermediate variables.
        """
        assert(output_name not in self._node_dict)
        # All variables needed to compute the final output.
        input_vars: Set[str] = set()
        # All arrays needed to compute the final output.
        input_arrays: Set[str] = set()
        # All MathOperations in the node that computes the final output
        node_mathop_list: List[MathOperation] = []
        # Find all mathops that compute the output array as well as all their input dependencies.
        for mathop in reversed(self._mathops_list):
            mathop_output: str = mathop.output_var_name
            if mathop_output == output_name or mathop_output in input_vars:
                node_mathop_list.append(mathop)
                if mathop.first_input_var is not None:
                    input_vars.add(mathop.first_input_var)
                if mathop.first_input_index is not None:
                    input_arrays.add(mathop.first_input_index[0])
                if mathop.second_input_var is not None:
                    input_vars.add(mathop.second_input_var)
                if mathop.second_input_index is not None:
                    input_arrays.add(mathop.second_input_index[0])
            elif mathop_output in input_arrays:
                if mathop_output not in self._node_dict:
                    node: Optional[TransformMathNode] =\
                        self._build_math_transform_for_output(mathop_output)
                    if node is None:
                        panic("No way to build {0} is found".format(mathop_output))
        # Return None if the final output is not buildable.
        if len(node_mathop_list) == 0:
            return None
        # The list was built backwards, reverse it.
        node_mathop_list.reverse()
        # Build the final node to return.
        input_nodes: List[WillumpGraphNode] =\
            list(map(lambda array: self._node_dict[array], input_arrays))
        return_node: TransformMathNode = TransformMathNode(input_nodes,
                                                           node_mathop_list, output_name)
        self._node_dict[output_name] = return_node
        return return_node

    @staticmethod
    def _pybinop_to_wbinop(binop: ast.operator) -> str:
        """
        Convert from AST binops to strings for TransformMathNode.
        """
        if isinstance(binop, ast.Add):
            return "+"
        elif isinstance(binop, ast.Sub):
            return "-"
        elif isinstance(binop, ast.Mult):
            return "*"
        elif isinstance(binop, ast.Div):
            return "/"
        else:
            return ""
