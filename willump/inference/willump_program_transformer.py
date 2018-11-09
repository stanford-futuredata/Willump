import ast
import copy
from typing import List

import willump.evaluation.evaluator as weval
from willump.inference.python_to_graph import WillumpGraphBuilder


class WillumpProgramTransformer(ast.NodeTransformer):
    """
    Transforms a Python program to replace all calls to a target featurization function
    with calls to equivalent Weld.
    """

    weld_program: str
    target_function: str

    def __init__(self, weld_program: str, target_function: str) -> None:
        self.weld_program = weld_program
        self.target_function = target_function

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """
        Replace all calls to target_function with calls to execute weld_program.
        in Weld.
        """
        called_function_name: str = WillumpGraphBuilder._get_function_name(node)
        new_node = copy.deepcopy(node)
        if called_function_name == self.target_function:
            # Define a func for a call into weval.evaluate_weld
            weld_eval_func: ast.Attribute = ast.Attribute()
            weld_eval_func.attr = "evaluate_weld"
            weld_eval_func.ctx = ast.Load()
            weld_eval_func_value: ast.Name = ast.Name()
            weld_eval_func_value.id = "weval"
            weld_eval_func_value.ctx = ast.Load()
            weld_eval_func.value = weld_eval_func_value

            weld_program_node = ast.Str()
            weld_program_node.s = "{0}".format(self.weld_program)

            new_node.args.insert(0, weld_program_node)
            new_node.func = weld_eval_func
            return ast.copy_location(new_node, node)
        else:
            transformed_arg_list: List[ast.expr] = []
            for arg in node.args:
                transformed_arg_list.append(self.visit(arg))
            new_node.args = transformed_arg_list
            return ast.copy_location(new_node, node)


