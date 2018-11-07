import ast
from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.inference.willump_graph_builder import WillumpGraphBuilder


class FirstFunctionDefFinder(ast.NodeVisitor):
    """
    Get the AST node for the first function in a Python AST.
    """
    function_def: ast.FunctionDef

    def visit_FunctionDef(self, node):
        self.function_def = node


def infer_graph(input_python: str) -> WillumpGraph:
    python_ast = ast.parse(input_python, "exec")
    first_fundef_finder = FirstFunctionDefFinder()
    first_fundef_finder.visit(python_ast)
    python_fundef: ast.FunctionDef = first_fundef_finder.function_def

    graph_builder = WillumpGraphBuilder()
    graph_builder.visit(python_fundef)

    return graph_builder.get_willump_graph()
