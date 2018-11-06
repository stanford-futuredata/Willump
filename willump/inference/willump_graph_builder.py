import ast

from willump import panic
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_graph_node import WillumpGraphNode
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.transform_math_node import TransformMathNode, MathOperation, MathOperationInput

from typing import MutableMapping, List, Tuple, Optional


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from raw Python.  Assumptions we currently make:

    1.  Input Python contains a single function, from which the graph shall be extracted.

    2.  This function takes in and returns a numpy.float64 array.
    """
    willump_graph: WillumpGraph
    # A map from the names of variables to the nodes that generate them.
    _node_dict: MutableMapping[str, WillumpGraphNode] = {}
    _mathops_list: List[MathOperation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in node.args.args:
            arg_name: str = arg.arg
            input_node: WillumpInputNode = WillumpInputNode(arg_name)
            self._node_dict[arg_name] = input_node
        ast.NodeVisitor.generic_visit(self, node)

    def _analyze_assignment_target(self, node: ast.expr) -> Tuple[str, bool]:
        """
        Return the name and type of the target of an assignment statement.
        :param node: An assignment target.
        :return: The name of the variable being assigned to and whether it is a vector.
        """
        if isinstance(node, ast.Name):
            return node.id, False
        elif isinstance(node, ast.Subscript):
            node_name, _ = self._analyze_assignment_target(node.value)
            return node_name, True
        else:
            panic("Unrecognized assignment to: {0}".format(ast.dump(node)))
            return "", False

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

    def _expr_to_math_operation_input(self, expr: ast.expr) -> MathOperationInput:
        """
        Convert an expression input to a binary or unary operation into a MathOperationInput.
        """
        if isinstance(expr, ast.Num):
            return MathOperationInput(input_literal=expr.n)
        elif isinstance(expr, ast.Subscript):
            return MathOperationInput(input_index=(expr.value.id, expr.slice.value.n))
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

        TODO:  Proper input handling.
        """
        return TransformMathNode(self._node_dict["input_numpy_array"],
                                 self._mathops_list, output_name)

    def visit_Assign(self, node: ast.Assign):
        # Assume assignment to only one variable for now.
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
