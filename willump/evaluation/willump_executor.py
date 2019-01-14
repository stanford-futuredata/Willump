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

from typing import Callable, MutableMapping, Mapping, List, Tuple
import typing

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

version_number = 0


def compile_weld_program(weld_program: str, type_map: Mapping[str, WeldType],
                         aux_data=None, base_filename="weld_llvm_caller") -> str:
    """
    Compile a Weld program to LLVM, then compile this into a Python C extension in a shared object
    file which can run the Weld program from Python.

    Return the name of the generated module.  Each Weld module must have a unique name because
    Python cannot reload shared libraries once loaded.

    Import the new module with:
    importlib.import_module(module_name)
    """
    if aux_data is None:
        aux_data = []
    global version_number
    # Compile the Weld program to LLVM and dump the LLVM.
    willump_home: str = os.environ["WILLUMP_HOME"]
    willump_build_dir: str = os.path.join(willump_home, "build")
    llvm_dump_name: str = "{0}{1}.ll".format(base_filename, version_number)
    llvm_dump_location = os.path.join(willump_build_dir, llvm_dump_name)
    if os.path.isfile(llvm_dump_location):
        os.remove(llvm_dump_location)
    weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
    weld_object.weld_code = weld_program
    input_types: List[WeldType] = []
    arg_index: int = 0
    while "__willump_arg{0}".format(arg_index) in type_map:
        input_types.append(type_map["__willump_arg{0}".format(arg_index)])
        arg_index += 1
    for (_, type_name) in aux_data:
        input_types.append(type_name)
    weld_object.willump_dump_llvm(input_types, input_directory=willump_build_dir,
                                  input_filename=llvm_dump_name)

    # Compile the LLVM to assembly and build the C++ glue code with it.
    driver_file = generate_cpp_driver(version_number, type_map, base_filename, aux_data)
    output_filename: str = os.path.join(willump_build_dir,
                                        "{0}{1}.so".format(base_filename, version_number))
    # TODO:  Make this call more portable.
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I{0}".format(sysconfig.get_path("include")),
                    "-I{0}".format(numpy.get_include()),
                    "-o", output_filename,
                    driver_file, llvm_dump_location])
    version_number += 1
    importlib.invalidate_caches()
    return "{0}{1}".format(base_filename, version_number - 1)


willump_typing_map_set: MutableMapping[str, MutableMapping[str, WeldType]] = {}
willump_static_vars_set: MutableMapping[str, object] = {}


def py_weld_program_to_statements(python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], List[str]]]],
                                  aux_data: List[Tuple[int, WeldType]], type_map: MutableMapping[str, WeldType]) \
        -> Tuple[List[ast.AST], List[str]]:
    python_weld_program = list(map(lambda x: (willump.evaluation.willump_weld_generator.set_input_names(x[0], x[1],
                                                                                                        aux_data), x[1],
                                              x[2]) if isinstance(x, tuple) else x, python_weld_program))
    type_map = copy.copy(type_map)
    arg_index: int = 0
    while "__willump_arg{0}".format(arg_index) in type_map:
        del type_map["__willump_arg{0}".format(arg_index)]
        arg_index += 1
    del type_map["__willump_retval"]
    all_python_program: List[ast.AST] = []
    module_to_import = []
    for entry in python_weld_program:
        if isinstance(entry, ast.AST):
            all_python_program.append(entry)
        else:
            weld_program, input_list, output_names = entry
            entry_type_map = copy.copy(type_map)
            for i, input_name in enumerate(input_list):
                entry_type_map["__willump_arg{0}".format(i)] = entry_type_map[input_name]
            for i, output_name in enumerate(output_names):
                entry_type_map["__willump_retval{0}".format(i)] = entry_type_map[output_name]
            module_name = compile_weld_program(weld_program, entry_type_map, aux_data)
            module_to_import.append(module_name)
            argument_string = ""
            for input_name in input_list:
                argument_string += input_name + ","
            argument_string = argument_string[:-1]  # Remove trailing comma.
            output_string = ""
            for output_name in output_names:
                output_string += output_name + ","
            output_string = output_string[:-1]  # Remove trailing comma.
            python_string = "%s = %s.caller_func(%s)" % (output_string, module_name, argument_string)
            python_ast_module: ast.Module = ast.parse(python_string)
            all_python_program = all_python_program + python_ast_module.body
    return all_python_program, module_to_import


def py_weld_statements_to_ast(py_weld_statements: List[ast.AST],
                              original_functiondef: ast.Module) -> ast.Module:
    new_functiondef = copy.copy(original_functiondef)
    new_functiondef.body[0].body = py_weld_statements
    new_functiondef.body[0].decorator_list = []
    new_functiondef = ast.fix_missing_locations(new_functiondef)
    return new_functiondef


def willump_execute(batch=True) -> Callable:
    """
    Decorator for a Python function that executes the function using Willump.

    Makes assumptions about the input Python outlined in WillumpGraphBuilder.

    Batch is true if batch-executing (multiple inputs and outputs per call) and
    false if point-executing (one input and output per call).
    """
    def willump_execute_inner(func: Callable) -> Callable:
        from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
        from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
        from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
        llvm_runner_func: str = "llvm_runner_func{0}".format(version_number)
        globals()[llvm_runner_func] = None

        def wrapper(*args):
            if globals()[llvm_runner_func] is None:
                if llvm_runner_func not in willump_typing_map_set:
                    # On the first run of the function, instrument it to construct a map from variables
                    # in the function to their types at runtime.
                    willump_typing_map_set[llvm_runner_func] = {}
                    willump_static_vars_set[llvm_runner_func] = {}
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
                    augmented_globals["willump_static_vars"] = willump_static_vars_set[llvm_runner_func]
                    augmented_globals["py_var_to_weld_type"] = py_var_to_weld_type
                    # Run the instrumented function.
                    exec(compile(new_ast, filename="<ast>", mode="exec"), augmented_globals,
                         local_namespace)
                    return local_namespace[function_name](*args)
                else:
                    # With the types of variables all known, we can compile the function.  First,
                    # infer the Willump graph from the function's AST.
                    willump_typing_map: Mapping[str, WeldType] = \
                        willump_typing_map_set[llvm_runner_func]
                    willump_static_vars: Mapping[str, object] = \
                        willump_static_vars_set[llvm_runner_func]
                    python_source = inspect.getsource(func)
                    python_ast: ast.Module = ast.parse(python_source)
                    function_name: str = python_ast.body[0].name
                    graph_builder = WillumpGraphBuilder(willump_typing_map, willump_static_vars, batch=batch)
                    graph_builder.visit(python_ast)
                    python_graph: WillumpGraph = graph_builder.get_willump_graph()
                    aux_data: List[Tuple[int, WeldType]] = graph_builder.get_aux_data()
                    # Transform the Willump graph into blocks of Weld and Python code.  Compile the Weld blocks.
                    python_weld_program: List[typing.Union[ast.AST, Tuple[str, List[str], str]]] = \
                        willump.evaluation.willump_weld_generator.graph_to_weld(python_graph)
                    python_statement_list, modules_to_import = py_weld_program_to_statements(python_weld_program,
                                                                                             aux_data, willump_typing_map)
                    compiled_functiondef = py_weld_statements_to_ast(python_statement_list, python_ast)
                    augmented_globals = copy.copy(func.__globals__)
                    # Import all of the compiled Weld blocks called from the transformed function.
                    for module in modules_to_import:
                        augmented_globals[module] = importlib.import_module(module)
                    local_namespace = {}
                    # Call the transformed function with its original arguments.
                    exec(compile(compiled_functiondef, filename="<ast>", mode="exec"), augmented_globals,
                         local_namespace)
                    globals()[llvm_runner_func] = local_namespace[function_name]
                    return globals()[llvm_runner_func](*args)
            else:
                # Run the compiled function.
                return globals()[llvm_runner_func](*args)

        return wrapper
    return willump_execute_inner
