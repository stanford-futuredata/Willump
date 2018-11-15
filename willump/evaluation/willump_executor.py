import ast
import numpy
import os
import subprocess
import importlib
import sysconfig
import inspect

import weld.weldobject
from weld.types import *
import weld.encoders

import willump.evaluation.willump_weld_generator
from willump.graph.willump_graph import WillumpGraph
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder

from typing import Callable

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

version_number = 0


def generate_cpp_file(file_version: int) -> str:
    """
    Generate versioned CPP files for each run of Willump.

    TODO:  Do C++ code generation to allow multiple inputs and different input types.
    """
    willump_home: str = os.environ["WILLUMP_HOME"]
    with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller.cpp"), "r") as infile:
        buffer = infile.read()
        new_function_name = "weld_llvm_caller{0}".format(file_version)
        buffer = buffer.replace("weld_llvm_caller", new_function_name)
        new_file_name = os.path.join(willump_home, "build",
                                     "weld_llvm_caller{0}.cpp".format(file_version))
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
    willump_home: str = os.environ["WILLUMP_HOME"]
    caller_file: str = generate_cpp_file(version_number)
    output_filename: str = os.path.join(willump_home, "build",
                                        "weld_llvm_caller{0}.so".format(version_number))
    # TODO:  Make this call more portable.
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I{0}".format(sysconfig.get_path("include")),
                    "-I{0}".format(numpy.get_include()),
                    "-o", output_filename,
                    caller_file, "code-llvm-opt.ll"])
    os.remove("code-llvm-opt.ll")
    version_number += 1
    importlib.invalidate_caches()
    return "weld_llvm_caller{0}".format(version_number - 1)


def willump_execute(func: Callable) -> Callable:
    """
    Decorator for a Python function that executes the function using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    TODO:  Make those assumptions weaker.
    """
    python_source = inspect.getsource(func)
    python_ast: ast.AST = ast.parse(python_source)
    graph_builder = WillumpGraphBuilder()
    graph_builder.visit(python_ast)
    python_graph: WillumpGraph = graph_builder.get_willump_graph()
    python_weld: str = willump.evaluation.willump_weld_generator.graph_to_weld(python_graph)

    module_name = compile_weld_program(python_weld)
    weld_llvm_caller = importlib.import_module(module_name)
    new_func: Callable = weld_llvm_caller.caller_func
    return new_func


def _infer_graph(input_python: str) -> WillumpGraph:
    """
    Infer the Willump graph of a Python program.

    For unit tests only!  Real Willump uses a decorator.
    """
    class FirstFunctionDefFinder(ast.NodeVisitor):
        """
        Get the AST node for the first function in a Python AST.
        """
        function_def: ast.FunctionDef

        def visit_FunctionDef(self, node):
            self.function_def = node

    python_ast = ast.parse(input_python, "exec")
    first_fundef_finder = FirstFunctionDefFinder()
    first_fundef_finder.visit(python_ast)
    python_fundef: ast.FunctionDef = first_fundef_finder.function_def

    graph_builder = WillumpGraphBuilder()
    graph_builder.visit(python_fundef)

    return graph_builder.get_willump_graph()

