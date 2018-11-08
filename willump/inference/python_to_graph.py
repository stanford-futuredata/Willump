import ast

from willump.graph.willump_graph import WillumpGraph
from willump.inference.willump_graph_builder import WillumpGraphBuilder
from willump.inference.willump_program_transformer import WillumpProgramTransformer

import willump.evaluation.evaluator as weval


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


def willump_execute_python(input_python: str) -> None:
    """
    Execute a Python program using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    TODO:  Make those assumptions weaker.
    """
    python_ast: ast.AST = ast.parse(input_python)
    python_graph: WillumpGraph = infer_graph(input_python)
    python_weld: str = weval.graph_to_weld(python_graph)
    graph_transformer: WillumpProgramTransformer = WillumpProgramTransformer(python_weld, "process_row")
    python_ast = graph_transformer.visit(python_ast)
    ast.fix_missing_locations(python_ast)
    print(ast.dump(python_ast))
    exec(compile(python_ast, filename="<ast>", mode="exec"), globals(), globals())

