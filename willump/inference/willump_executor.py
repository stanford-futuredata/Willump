import ast
import numpy
import os
import subprocess
import time

import weld.weldobject
from weld.types import *
import weld.encoders

from willump.graph.willump_graph import WillumpGraph
from willump.inference.willump_graph_builder import WillumpGraphBuilder
from willump.inference.willump_program_transformer import WillumpProgramTransformer

import willump.evaluation.evaluator as weval

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

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


def compile_weld_program(weld_program: str) -> None:
    """
    Compile a Weld program to LLVM, then compile this into a Python C extension in a shared object
    file which can run the Weld program from Python.

    After this function executes, this caller can be imported using the following:

    from weld_llvm_caller import weld_llvm_caller
    """
    # Compile the Weld program to LLVM and dump the LLVM.
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    weld_object.weld_code = weld_program.format("_inp0")
    weld_object.willump_dump_llvm([WeldVec(WeldDouble())])

    # Compile the LLVM to assembly and build the C++ glue code with it.
    if not os.path.exists("build"):
        os.mkdir("build")
    # TODO:  Make this call more portable.
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I/usr/include/python3.6", "-I{0}".format(numpy.get_include()),
                    "-o", "build/weld_llvm_caller.so",
                    "cppextensions/weld_llvm_caller.cpp", "code-llvm-opt.ll"])


def willump_execute_python(input_python: str) -> None:
    """
    Execute a Python program using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    TODO:  Make those assumptions weaker.
    """
    python_ast: ast.AST = ast.parse(input_python)
    python_graph: WillumpGraph = infer_graph(input_python)
    python_weld: str = weval.graph_to_weld(python_graph)

    compile_weld_program(python_weld)
    from weld_llvm_caller import weld_llvm_caller
    global weld_llvm_caller

    graph_transformer: WillumpProgramTransformer = WillumpProgramTransformer(python_weld,
                                                                             "process_row")
    python_ast = graph_transformer.visit(python_ast)
    ast.fix_missing_locations(python_ast)
    exec(compile(python_ast, filename="<ast>", mode="exec"), globals(), globals())
    os.remove("code-llvm-opt.ll")


