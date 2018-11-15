import ast
import copy
from typing import List

from willump.evaluation.willump_graph_builder import WillumpGraphBuilder


class WillumpRuntimeTypeDiscovery(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_node = copy.deepcopy(node)
        new_body: List[ast.AST] = []
        for body_entry in node.body:
            new_body.append(body_entry)
            if isinstance(body_entry, ast.Assign):
                assert(len(body_entry.targets) == 1)  # Assume assignment to only one variable.
                target: ast.expr = body_entry.targets[0]
                target_type_statement: List[ast.AST] = self._analyze_target_type(target)
                new_body = new_body + target_type_statement
        new_node.body = new_body
        # No recursion allowed!
        new_node.decorator_list = []
        return ast.copy_location(new_node, node)

    @staticmethod
    def _analyze_target_type(target: ast.expr) -> List[ast.AST]:
        target_name: str = WillumpGraphBuilder.get_assignment_target_name(target)
        target_analysis_instrumentation_code: str = \
        """willump_typing_map["{0}"] = type({0})""".format(target_name)
        instrumentation_ast: ast.Module = ast.parse(target_analysis_instrumentation_code, "exec")
        instrumentation_statements: List[ast.AST] = instrumentation_ast.body
        return instrumentation_statements

