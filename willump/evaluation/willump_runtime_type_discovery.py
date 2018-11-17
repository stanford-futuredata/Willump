import ast
import copy
import numpy
from typing import List

from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump import panic

from weld.types import *


class WillumpRuntimeTypeDiscovery(ast.NodeTransformer):
    """
    Annotate the AST of a Python function to record the Weld type of every variable after it is
    assigned. This must run in a global namespace that contains a willump_typing_map variable
    (into which the types will be recorded) as well as the py_var_to_weld_type function.

    TODO:  Add support for control flow changes inside the function body.
    """
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_node = copy.deepcopy(node)
        new_body: List[ast.stmt] = []
        for i, arg in enumerate(node.args.args):
            # First, type the arguments.
            argument_name: str = arg.arg
            argument_internal_name = "__willump_arg{0}".format(i)
            argument_instrumentation_code: str = \
            """willump_typing_map["{0}"] = py_var_to_weld_type({0})\n""".format(argument_name) +\
            """willump_typing_map["{0}"] = py_var_to_weld_type({1})"""\
                .format(argument_internal_name, argument_name)
            instrumentation_ast: ast.Module = ast.parse(argument_instrumentation_code, "exec")
            instrumentation_statements: List[ast.stmt] = instrumentation_ast.body
            new_body = new_body + instrumentation_statements
        for body_entry in node.body:
            if isinstance(body_entry, ast.Assign):
                # Type all variables as they are assigned.
                new_body.append(body_entry)
                assert(len(body_entry.targets) == 1)  # Assume assignment to only one variable.
                target: ast.expr = body_entry.targets[0]
                target_type_statement: List[ast.stmt] = self._analyze_target_type(target)
                new_body = new_body + target_type_statement
            elif isinstance(body_entry, ast.Return):
                # Type the function's return value.
                new_assignment: ast.Assign = ast.Assign()
                new_assignment_target: ast.Name = ast.Name()
                new_assignment_target.id = "__willump_retval"
                new_assignment_target.ctx = ast.Store()
                new_assignment.targets = [new_assignment_target]
                new_assignment.value = body_entry.value
                new_body.append(new_assignment)
                return_type_statement: List[ast.stmt] = \
                    self._analyze_target_type(new_assignment_target)
                new_body = new_body + return_type_statement
                new_body.append(body_entry)
        new_node.body = new_body
        # No recursion allowed!
        new_node.decorator_list = []
        return ast.copy_location(new_node, node)

    @staticmethod
    def _analyze_target_type(target: ast.expr) -> List[ast.stmt]:
        """
        Create a statement from the target of an assignment that will insert into a global
        dict the type of the target.
        """
        target_name: str = WillumpGraphBuilder.get_assignment_target_name(target)
        target_analysis_instrumentation_code: str = \
        """willump_typing_map["{0}"] = py_var_to_weld_type({0})""".format(target_name)
        instrumentation_ast: ast.Module = ast.parse(target_analysis_instrumentation_code, "exec")
        instrumentation_statements: List[ast.stmt] = instrumentation_ast.body
        return instrumentation_statements


def py_var_to_weld_type(py_var: object) -> WeldType:
    """
    Get the Weld type of a Python variable.

    TODO:  Handle more types of variables.
    """
    if isinstance(py_var, int):
        return WeldInt()
    elif isinstance(py_var, float):
        return WeldDouble()
    # TODO:  Handle multidimensional arrays
    elif isinstance(py_var, numpy.ndarray):
        if py_var.dtype == numpy.int8:
            return WeldVec(WeldChar())
        elif py_var.dtype == numpy.int16:
            return WeldVec(WeldInt16())
        elif py_var.dtype == numpy.int32:
            return WeldVec(WeldInt())
        elif py_var.dtype == numpy.int64:
            return WeldVec(WeldLong())
        elif py_var.dtype == numpy.float32:
            return WeldVec(WeldFloat())
        elif py_var.dtype == numpy.float64:
            return WeldVec(WeldDouble())
        else:
            panic("Unrecognized ndarray type {0}".format(py_var.dtype.__str__))
            return WeldType()
    else:
        panic("Unrecognized type of object {0} with type {1}"
              .format(py_var.__str__(), type(py_var)))
        return WeldType()
