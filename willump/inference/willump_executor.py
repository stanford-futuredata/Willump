import ast
import numpy
import os
import subprocess
import ctypes
import importlib

import weld.weldobject
from weld.types import *
import weld.encoders

from willump.graph.willump_graph import WillumpGraph
from willump.inference.willump_graph_builder import WillumpGraphBuilder
from willump.inference.willump_program_transformer import WillumpProgramTransformer

import willump.evaluation.evaluator as weval

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

version_number = 0

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


def generate_cpp_file(file_version: int) -> str:
    """
    Generate versioned CPP files for each run of Willump.
    """
    with open("cppextensions/weld_llvm_caller.cpp", "r") as infile:
        buffer = infile.read()
        new_function_name = "weld_llvm_caller{0}".format(file_version)
        buffer = buffer.replace("weld_llvm_caller", new_function_name)
        new_file_name = "build/weld_llvm_caller{0}.cpp".format(file_version)
        with open(new_file_name, "w") as outfile:
            outfile.write(buffer)
    return new_file_name


def compile_weld_program(weld_program: str) -> str:
    """
    Compile a Weld program to LLVM, then compile this into a Python C extension in a shared object
    file which can run the Weld program from Python.

    Return the name of the generated module.  Each Weld module must have a unique name because
    Python cannot reload shared libraries once loaded.

    Import the new module with:
    importlib.import_module(module_name)
    """
    global version_number
    # Compile the Weld program to LLVM and dump the LLVM.
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    weld_object.weld_code = weld_program.format("_inp0")
    weld_object.willump_dump_llvm([WeldVec(WeldDouble())])

    # Compile the LLVM to assembly and build the C++ glue code with it.
    if not os.path.exists("build"):
        os.mkdir("build")
    caller_file = generate_cpp_file(version_number)
    # TODO:  Make this call more portable.
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I/usr/include/python3.6", "-I{0}".format(numpy.get_include()),
                    "-o", "build/weld_llvm_caller{0}.so".format(version_number),
                    caller_file, "code-llvm-opt.ll"])
    version_number += 1
    return "weld_llvm_caller{0}".format(version_number - 1)


def willump_execute_python(input_python: str) -> None:
    """
    Execute a Python program using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    TODO:  Make those assumptions weaker.
    """
    python_ast: ast.AST = ast.parse(input_python)
    python_graph: WillumpGraph = infer_graph(input_python)
    python_weld: str = weval.graph_to_weld(python_graph)

    module_name = compile_weld_program(python_weld)
    weld_llvm_caller = importlib.import_module(module_name)

    graph_transformer: WillumpProgramTransformer = WillumpProgramTransformer("process_row")
    python_ast = graph_transformer.visit(python_ast)
    ast.fix_missing_locations(python_ast)
    exec(compile(python_ast, filename="<ast>", mode="exec"), globals(), locals())
    os.remove("code-llvm-opt.ll")


