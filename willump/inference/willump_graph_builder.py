import ast
from willump.graph.willump_graph import WillumpGraph


class WillumpGraphBuilder(ast.NodeVisitor):
    """
    Builds a Willump graph from raw Python.  Assumptions we currently make:

    1.  Input Python contains a single function, from which the graph shall be extracted.

    2.  This function takes in and returns a numpy.float64 array.
    """
    willump_graph: WillumpGraph

    def visit_FunctionDef(self, node: ast.FunctionDef):
        print(ast.dump(node.args))
        ast.NodeVisitor.generic_visit(self, node)

    def generic_visit(self, node):
        print(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def get_willump_graph(self) -> WillumpGraph:
        return self.willump_graph
