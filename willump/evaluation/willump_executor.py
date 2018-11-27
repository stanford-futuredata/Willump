import ast
import numpy
import os
import subprocess
import importlib
import sysconfig
import inspect
import copy

import weld.weldobject
from weld.types import *
import weld.encoders

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_driver_generator import generate_cpp_driver
from willump.graph.willump_graph import WillumpGraph
from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type

from typing import Callable, MutableMapping, Mapping

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

version_number = 0


def compile_weld_program(weld_program: str, type_map: Mapping[str, WeldType]) -> str:
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
    willump_home: str = os.environ["WILLUMP_HOME"]
    willump_build_dir: str = os.path.join(willump_home, "build")
    llvm_dump_name: str = "weld_llvm_caller{0}.ll".format(version_number)
    llvm_dump_location = os.path.join(willump_build_dir, llvm_dump_name)
    if os.path.isfile(llvm_dump_location):
        os.remove(llvm_dump_location)
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    weld_object.weld_code = weld_program.replace("WELD_INPUT_1", "_inp0")
    weld_object.willump_dump_llvm([type_map["__willump_arg0"]], input_directory=willump_build_dir,
                                  input_filename=llvm_dump_name)

    # Compile the LLVM to assembly and build the C++ glue code with it.
    caller_file: str = generate_cpp_driver(version_number, type_map)
    output_filename: str = os.path.join(willump_build_dir,
                                        "weld_llvm_caller{0}.so".format(version_number))
    # TODO:  Make this call more portable.
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I{0}".format(sysconfig.get_path("include")),
                    "-I{0}".format(numpy.get_include()),
                    "-o", output_filename,
                    caller_file, llvm_dump_location])
    version_number += 1
    importlib.invalidate_caches()
    return "weld_llvm_caller{0}".format(version_number - 1)


willump_typing_map_set: MutableMapping[str, MutableMapping[str, WeldType]] = {}


def willump_execute(func: Callable) -> Callable:
    """
    Decorator for a Python function that executes the function using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    TODO:  Make those assumptions weaker.
    """
    llvm_runner_func: str = "llvm_runner_func{0}".format(version_number)
    globals()[llvm_runner_func] = None

    def wrapper(*args):
        if globals()[llvm_runner_func] is None:
            if llvm_runner_func not in willump_typing_map_set:
                # On the first run of the function, instrument it to construct a map from variables
                # in the function to their types at runtime.
                willump_typing_map_set[llvm_runner_func] = {}
                python_source = inspect.getsource(func)
                python_ast: ast.AST = ast.parse(python_source)
                function_name: str = python_ast.body[0].name
                type_discover: WillumpRuntimeTypeDiscovery = WillumpRuntimeTypeDiscovery()
                # Create an instrumented AST that will fill willump_typing_map with the Weld types
                # of all variables in the function.
                new_ast: ast.AST = type_discover.visit(python_ast)
                new_ast = ast.fix_missing_locations(new_ast)
                local_namespace = {}
                # Create namespaces the instrumented function can run in containing both its
                # original globals and the ones the instrumentation needs.
                augmented_globals = copy.copy(func.__globals__)
                augmented_globals["willump_typing_map"] = willump_typing_map_set[llvm_runner_func]
                augmented_globals["py_var_to_weld_type"] = py_var_to_weld_type
                # Run the instrumented function.
                exec(compile(new_ast, filename="<ast>", mode="exec"), augmented_globals,
                     local_namespace)
                return local_namespace[function_name](args[0])
            else:
                # With the types of variables all known, we can compile the function.  First,
                # infer the Willump graph from the function's AST.
                willump_typing_map: Mapping[str, WeldType] = \
                    willump_typing_map_set[llvm_runner_func]
                python_source = inspect.getsource(func)
                python_ast: ast.AST = ast.parse(python_source)
                graph_builder = WillumpGraphBuilder(willump_typing_map)
                graph_builder.visit(python_ast)
                python_graph: WillumpGraph = graph_builder.get_willump_graph()
                # Convert the Willump graph to a Weld program.
                python_weld: str =\
                    willump.evaluation.willump_weld_generator.graph_to_weld(python_graph)
                # Compile the Weld to a Python C extension, then call into the extension.
                module_name = compile_weld_program(python_weld, willump_typing_map)
                weld_llvm_caller = importlib.import_module(module_name)
                new_func: Callable = weld_llvm_caller.caller_func
                globals()[llvm_runner_func] = new_func
                return globals()[llvm_runner_func](args[0])
        else:
            # Run the compiled function.
            return globals()[llvm_runner_func](args[0])
    return wrapper
