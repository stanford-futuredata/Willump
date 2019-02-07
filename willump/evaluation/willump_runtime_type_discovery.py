import ast
import copy
import numpy
from typing import List, Optional
import pandas.core.frame

from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump import *
import scipy.sparse.csr
from willump.willump_utilities import *

from weld.types import *


class WillumpRuntimeTypeDiscovery(ast.NodeTransformer):
    """
    Annotate the AST of a Python function to record the Weld type of every variable after it is
    assigned. This must run in a global namespace that contains a willump_typing_map variable
    (into which the types will be recorded) as well as the py_var_to_weld_type function.

    Also extract a list of values of "static variables"--important and unchanging values such as
    the weights of a logistic regression model or the contents of a vocabulary.

    TODO:  Add support for control flow changes inside the function body.
    """

    batch: bool

    def __init__(self, batch: bool = True):
        self.batch = batch

    def process_body(self, body):
        new_body: List[ast.stmt] = []
        for body_entry in body:
            if isinstance(body_entry, ast.Assign):
                # Type all variables as they are assigned.
                new_body.append(body_entry)
                assert (len(body_entry.targets) == 1)  # Assume assignment to only one variable.
                target: ast.expr = body_entry.targets[0]
                target_type_statement: List[ast.stmt] = self._analyze_target_type(target)
                new_body = new_body + target_type_statement
                # Remember static variables if present.
                value: ast.expr = body_entry.value
                extract_static_vars_statements = self._maybe_extract_static_variables(value)
                new_body = new_body + extract_static_vars_statements
            else:
                new_body.append(body_entry)
        return new_body

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        new_node = copy.deepcopy(node)
        new_body: List[ast.stmt] = []
        for i, arg in enumerate(node.args.args):
            # First, type the arguments.
            argument_name: str = arg.arg
            argument_instrumentation_code: str = \
                """willump_typing_map["{0}_{2}"] = py_var_to_weld_type({0}, {1})\n""" \
                    .format(argument_name, self.batch, node.lineno)
            instrumentation_ast: ast.Module = ast.parse(argument_instrumentation_code, "exec")
            instrumentation_statements: List[ast.stmt] = instrumentation_ast.body
            new_body = new_body + instrumentation_statements
        new_body += self.process_body(node.body)
        new_node.body = new_body
        # No recursion allowed!
        new_node.decorator_list = []
        self.generic_visit(new_node)
        return ast.copy_location(new_node, node)

    def visit_For(self, node: ast.For):
        new_node = copy.deepcopy(node)
        new_node.body = self.process_body(node.body)
        new_node.orelse = self.process_body(node.orelse)
        self.generic_visit(new_node)
        return ast.copy_location(new_node, node)

    def visit_If(self, node: ast.If):
        new_node = copy.deepcopy(node)
        new_node.body = self.process_body(node.body)
        new_node.orelse = self.process_body(node.orelse)
        self.generic_visit(new_node)
        return ast.copy_location(new_node, node)

    def visit_While(self, node: ast.While):
        new_node = copy.deepcopy(node)
        new_node.body = self.process_body(node.body)
        new_node.orelse = self.process_body(node.orelse)
        self.generic_visit(new_node)
        return ast.copy_location(new_node, node)

    def visit_With(self, node: ast.With):
        new_node = copy.deepcopy(node)
        new_node.body = self.process_body(node.body)
        self.generic_visit(new_node)
        return ast.copy_location(new_node, node)

    @staticmethod
    def _maybe_extract_static_variables(value: ast.expr) -> List[ast.stmt]:
        return_statements: List[ast.stmt] = []
        if isinstance(value, ast.Subscript):
            if isinstance(value.slice.value, ast.Name):
                index_name = value.slice.value.id
                static_variable_extraction_code = \
                    """willump_static_vars["{0}"] = {1}""" \
                        .format(WILLUMP_SUBSCRIPT_INDEX_NAME + str(value.lineno), index_name)
                index_name_instrumentation_ast: ast.Module = \
                    ast.parse(static_variable_extraction_code, "exec")
                index_name_instrumentation_statements: List[ast.stmt] = \
                    index_name_instrumentation_ast.body
                return_statements += index_name_instrumentation_statements
        elif isinstance(value, ast.Call):
            called_function_name: str = WillumpGraphBuilder._get_function_name(value)
            if "willump_frequency_count" in called_function_name:
                vocab_dict_name: str = value.args[1].id
                static_variable_extraction_code = \
                    """willump_static_vars["{0}"] = {1}""" \
                        .format(WILLUMP_FREQUENCY_COUNT_VOCAB, vocab_dict_name)
                freq_count_instrumentation_ast: ast.Module = \
                    ast.parse(static_variable_extraction_code, "exec")
                freq_count_instrumentation_statements: List[ast.stmt] = \
                    freq_count_instrumentation_ast.body
                return_statements += freq_count_instrumentation_statements
            elif "predict" in called_function_name:
                if isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                    model_name = value.func.value.id
                    static_variable_extraction_code = \
                        """if "sklearn.linear_model" in type({0}).__module__:\n""" \
                            .format(model_name) + \
                        """\twillump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_LINEAR_REGRESSION_WEIGHTS, model_name, "coef_") + \
                        """\twillump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_LINEAR_REGRESSION_INTERCEPT, model_name, "intercept_")
                    logit_instrumentation_ast: ast.Module = \
                        ast.parse(static_variable_extraction_code, "exec")
                    logit_instrumentation_statements: List[ast.stmt] = logit_instrumentation_ast.body
                    return_statements += logit_instrumentation_statements
            elif "transform" in called_function_name:
                if isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name):
                    lineno = str(value.lineno)
                    transformer_name = value.func.value.id
                    static_variable_extraction_code = \
                        """willump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_COUNT_VECTORIZER_VOCAB + lineno, transformer_name, "vocabulary_") + \
                        """willump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_COUNT_VECTORIZER_ANALYZER + lineno, transformer_name, "analyzer") + \
                        """willump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_COUNT_VECTORIZER_NGRAM_RANGE + lineno, transformer_name, "ngram_range") + \
                        """willump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_COUNT_VECTORIZER_LOWERCASE + lineno, transformer_name, "lowercase") + \
                        """if type({0}).__name__ == "TfidfVectorizer":\n""".format(transformer_name) + \
                        """\twillump_static_vars["{0}"] = {1}.{2}\n""" \
                            .format(WILLUMP_TFIDF_IDF_VECTOR + lineno, transformer_name, "idf_")
                    count_vectorizer_instrumentation_ast: ast.Module = \
                        ast.parse(static_variable_extraction_code, "exec")
                    count_vectorizer_instrumentation_statements: List[
                        ast.stmt] = count_vectorizer_instrumentation_ast.body
                    return_statements += count_vectorizer_instrumentation_statements
            elif "merge" in called_function_name:
                # TODO:  More robust extraction.
                static_variable_extraction_code = \
                    """willump_static_vars["{0}"] = {1}\n""" \
                        .format(WILLUMP_JOIN_RIGHT_DATAFRAME + str(value.lineno), value.args[0].id) + \
                    """willump_static_vars["{0}"] = {1}.dtypes\n""" \
                        .format(WILLUMP_JOIN_LEFT_DTYPES + str(value.lineno), value.func.value.id) + \
                    """willump_static_vars["{0}"] = "{1}"\n""" \
                        .format(WILLUMP_JOIN_HOW + str(value.lineno), value.keywords[0].value.s)
                if isinstance(value.keywords[1].value, ast.Name):
                    static_variable_extraction_code += \
                        """willump_static_vars["{0}"] = {1}\n""" \
                            .format(WILLUMP_JOIN_COL + str(value.lineno), value.keywords[1].value.id)
                else:
                    static_variable_extraction_code += \
                        """willump_static_vars["{0}"] = "{1}"\n""" \
                            .format(WILLUMP_JOIN_COL + str(value.lineno), value.keywords[1].value.s)
                join_instrumentation_ast: ast.Module = \
                    ast.parse(static_variable_extraction_code, "exec")
                join_instrumentation_statements: List[ast.stmt] = join_instrumentation_ast.body
                return_statements += join_instrumentation_statements

        return return_statements

    def _analyze_target_type(self, target: ast.expr) -> List[ast.stmt]:
        """
        Create a statement from the target of an assignment that will insert into a global
        dict the type of the target.
        """
        target_name: str = WillumpGraphBuilder.get_assignment_target_name(target)
        target_analysis_instrumentation_code: str = \
            """willump_typing_map["{0}_{2}"] = py_var_to_weld_type({0}, {1})""".format(target_name, self.batch,
                                                                                       target.lineno)
        instrumentation_ast: ast.Module = ast.parse(target_analysis_instrumentation_code, "exec")
        instrumentation_statements: List[ast.stmt] = instrumentation_ast.body
        return instrumentation_statements


def py_var_to_weld_type(py_var: object, batch) -> Optional[WeldType]:
    """
    Get the Weld type of a Python variable.

    TODO:  Handle more types of variables.
    """
    if isinstance(py_var, int):
        return WeldLong()
    elif isinstance(py_var, float):
        return WeldDouble()
    elif isinstance(py_var, str):
        return WeldStr()
    # TODO:  Find a more robust way to handle list types, this fails badly if the input is degenerate.
    elif isinstance(py_var, list) and len(py_var) > 0 and isinstance(py_var[0], str):
        return WeldVec(WeldStr())
    # Sparse matrix type used by CountVectorizer
    elif isinstance(py_var, scipy.sparse.csr.csr_matrix):
        if py_var.dtype == numpy.int8:
            return WeldCSR(WeldChar())
        elif py_var.dtype == numpy.int16:
            return WeldCSR(WeldInt16())
        elif py_var.dtype == numpy.int32:
            return WeldCSR(WeldInt())
        elif py_var.dtype == numpy.int64:
            return WeldCSR(WeldLong())
        elif py_var.dtype == numpy.float32:
            return WeldCSR(WeldFloat())
        elif py_var.dtype == numpy.float64:
            return WeldCSR(WeldDouble())
        else:
            panic("Unrecognized ndarray type {0}".format(py_var.dtype.__str__()))
            return None
    elif isinstance(py_var, pandas.core.frame.DataFrame):
        df_col_weld_types = []
        for dtype in py_var.dtypes:
            col_weld_type: WeldType = numpy_type_to_weld_type(dtype)
            if batch:
                df_col_weld_types.append(WeldVec(col_weld_type))
            else:
                df_col_weld_types.append(col_weld_type)
        return WeldPandas(df_col_weld_types, list(py_var.columns))
    elif isinstance(py_var, numpy.ndarray):
        if py_var.ndim > 1:
            return WeldVec(py_var_to_weld_type(py_var[0], batch))
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
        elif py_var.dtype == numpy.object:
            return WeldVec(py_var_to_weld_type(py_var[0], batch))
        else:
            panic("Unrecognized ndarray type {0}".format(py_var.dtype.__str__()))
            return None
    else:
        # print("Unrecognized var type {0}".format(type(py_var)))
        return None
