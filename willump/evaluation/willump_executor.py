import ast
import concurrent.futures
import copy
import importlib
import inspect
import subprocess
import sysconfig
import typing
from typing import MutableMapping, Mapping, List, Tuple

import weld.encoders
import weld.weldobject

import willump.evaluation.willump_weld_generator
from willump.evaluation.willump_driver_generator import generate_cpp_driver
from willump.evaluation.willump_runtime_code import *
from willump.graph.willump_graph import WillumpGraph
from willump.willump_utilities import *

_encoder = weld.encoders.NumpyArrayEncoder()
_decoder = weld.encoders.NumpyArrayDecoder()

version_number = 0


def compile_weld_program(weld_programs: typing.Union[str, List[str]], type_map: Mapping[str, WeldType],
                         input_names: List[str],
                         output_names: typing.Union[List[str], List[List[str]]],
                         aux_data=None, base_filename="weld_llvm_caller",
                         thread_runner_pointer=0) -> str:
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
    if isinstance(weld_programs, str):
        weld_programs = [weld_programs]
        output_names = [output_names]
    assert (len(weld_programs) == len(output_names))
    global version_number
    # Compile the Weld program to LLVM and dump the LLVM.
    willump_home: str = os.environ["WILLUMP_HOME"]
    willump_build_dir: str = os.path.join(willump_home, "build")
    llvm_dump_names: List[str] = []
    entry_point_names: List[str] = []
    for i in range(len(weld_programs)):
        llvm_dump_names.append("%s_%d_%d.ll" % (base_filename, version_number, i))
        entry_point_names.append("weld_run_%d_%d" % (version_number, i))
    llvm_dump_location_list = []
    for llvm_dump_name in llvm_dump_names:
        llvm_dump_location = os.path.join(willump_build_dir, llvm_dump_name)
        if os.path.isfile(llvm_dump_location):
            os.remove(llvm_dump_location)
        llvm_dump_location_list.append(llvm_dump_location)
    input_types: List[WeldType] = list(map(lambda name: type_map[name], input_names))
    for (_, type_name) in aux_data:
        input_types.append(type_name)
    for (weld_program, llvm_dump_name, entry_point_name) in zip(weld_programs, llvm_dump_names, entry_point_names):
        weld_object = weld.weldobject.WeldObject(_encoder, _decoder)
        weld_object.weld_code = weld_program
        weld_object.willump_dump_llvm(input_types, input_directory=willump_build_dir,
                                      input_filename=llvm_dump_name, entry_point=entry_point_name)
    # Compile the LLVM to assembly and build the C++ glue code with it.
    driver_file = generate_cpp_driver(version_number, type_map, input_names, output_names,
                                      base_filename, aux_data, thread_runner_pointer, entry_point_names)
    output_filename: str = os.path.join(willump_build_dir,
                                        "{0}{1}.so".format(base_filename, version_number))
    subprocess.run(["clang++", "-fPIC", "--shared", "-lweld", "-g", "-std=c++11", "-O3",
                    "-I{0}".format(sysconfig.get_path("include")),
                    "-I{0}".format(numpy.get_include()),
                    "-o", output_filename,
                    driver_file] + llvm_dump_location_list)
    version_number += 1
    importlib.invalidate_caches()
    return "{0}{1}".format(base_filename, version_number - 1)


willump_typing_map_set: MutableMapping[str, MutableMapping[str, WeldType]] = {}
willump_static_vars_set: MutableMapping[str, object] = {}


def py_weld_program_to_statements(
        python_weld_program: List[typing.Union[ast.AST, Tuple[List[str], List[str], List[List[str]]]]],
        aux_data: List[Tuple[int, WeldType]], type_map: MutableMapping[str, WeldType],
        num_workers) \
        -> Tuple[List[ast.AST], List[str]]:
    """
    Take in a list of mixed Python and Weld code.  Compile all Weld code into Python C Extensions.  Return
    the body of a valid Python function that calls all the Python and Weld code.  Return a list of the names
    of the Python C extensions containing the compiled Weld code.
    """

    def set_weld_statement_input_names(x: Tuple[List[str], List[str], List[List[str]]]):
        weld_strings, x_input_names, x_output_names = x
        updated_weld_strings = list(map(lambda weld_string: willump.evaluation.willump_weld_generator.set_input_names(
            weld_string, x_input_names, aux_data), weld_strings))
        return updated_weld_strings, x_input_names, x_output_names

    python_weld_program = list(map(lambda x: set_weld_statement_input_names(x)
    if isinstance(x, tuple) else x, python_weld_program))
    all_python_program: List[ast.AST] = []
    module_to_import = []

    # Make sure no entry was given more parallel jobs than we have threads (including main thread).
    for entry in python_weld_program:
        if isinstance(entry, tuple):
            weld_programs, _, _ = entry
            assert (num_workers + 1 >= len(weld_programs))
    # Create the thread pool.
    module_name = compile_weld_program("true", {}, [], [], base_filename="thread_pool_creator")
    thread_pool_creator = importlib.import_module(module_name)
    thread_runner_pointer = thread_pool_creator.caller_func(num_workers)

    for entry in python_weld_program:
        if isinstance(entry, ast.AST):
            all_python_program.append(entry)
        else:
            weld_programs, input_names, output_names_list = entry
            module_name = compile_weld_program(weld_programs, type_map, input_names=input_names,
                                               output_names=output_names_list, aux_data=aux_data,
                                               thread_runner_pointer=thread_runner_pointer)
            module_to_import.append(module_name)
            argument_string = ""
            for input_name in input_names:
                # Strip line numbers from variable names.
                argument_string += strip_linenos_from_var(input_name) + ","
            argument_string = argument_string[:-1]  # Remove trailing comma.
            output_names = [item for sublist in output_names_list for item in sublist]  # Flatten
            output_string = ""
            for output_name in output_names:
                # Strip line numbers from variable names.
                output_string += strip_linenos_from_var(output_name) + ","
            python_string = "%s = %s.caller_func(%s)" % (output_string, module_name, argument_string)
            python_ast_module: ast.Module = ast.parse(python_string)
            all_python_program = all_python_program + python_ast_module.body
    # all_python_program.insert(-5, ast.parse("print(cascading__more__train_data)").body[0])
    return all_python_program, module_to_import


def py_weld_statements_to_ast(py_weld_statements: List[ast.AST],
                              original_functiondef: ast.Module) -> ast.Module:
    """
    Take in the body of a valid Python function.  Return a Python AST that can be compiled into that program.
    """
    new_functiondef = copy.copy(original_functiondef)
    new_functiondef.body[0].body = py_weld_statements
    new_functiondef.body[0].decorator_list = []
    new_functiondef = ast.fix_missing_locations(new_functiondef)
    return new_functiondef


def willump_execute(batch=True, num_workers=0, async_funcs=(), training_cascades=None, eval_cascades=None,
                    cascade_threshold=1.0, cached_funcs=(), max_cache_size=None) -> Callable:
    """
    Decorator for a Python function that executes the function using Willump.
    """

    def willump_execute_inner(func: Callable) -> Callable:
        from willump.evaluation.willump_graph_builder import WillumpGraphBuilder
        from willump.evaluation.willump_runtime_type_discovery import WillumpRuntimeTypeDiscovery
        from willump.evaluation.willump_runtime_type_discovery import py_var_to_weld_type
        llvm_runner_func: str = "llvm_runner_func{0}".format(func.__name__)
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
                    willump_typing_map: MutableMapping[str, WeldType] = \
                        willump_typing_map_set[llvm_runner_func]
                    willump_static_vars: Mapping[str, object] = \
                        willump_static_vars_set[llvm_runner_func]
                    python_source = inspect.getsource(func)
                    python_ast: ast.Module = ast.parse(python_source)
                    function_name: str = python_ast.body[0].name
                    graph_builder = WillumpGraphBuilder(willump_typing_map, willump_static_vars, async_funcs, cached_funcs)
                    graph_builder.visit(python_ast)
                    python_graph: WillumpGraph = graph_builder.get_willump_graph()
                    aux_data: List[Tuple[int, WeldType]] = graph_builder.get_aux_data()
                    willump_cache_dict = {}
                    # Transform the Willump graph into blocks of Weld and Python code.  Compile the Weld blocks.
                    python_weld_program: List[typing.Union[ast.AST, Tuple[List[str], List[str], List[List[str]]]]] = \
                        willump.evaluation.willump_weld_generator.graph_to_weld(graph=python_graph,
                                                                                typing_map=willump_typing_map,
                                                                                training_cascades=training_cascades,
                                                                                eval_cascades=eval_cascades,
                                                                                cascade_threshold=cascade_threshold,
                                                                                aux_data=aux_data,
                                                                                willump_cache_dict=willump_cache_dict,
                                                                                max_cache_size=max_cache_size,
                                                                                batch=batch, num_workers=num_workers)
                    python_statement_list, modules_to_import = py_weld_program_to_statements(python_weld_program,
                                                                                             aux_data,
                                                                                             willump_typing_map,
                                                                                             num_workers=num_workers)
                    compiled_functiondef = py_weld_statements_to_ast(python_statement_list, python_ast)
                    # import astor
                    # print(astor.to_source(compiled_functiondef))
                    augmented_globals = copy.copy(func.__globals__)
                    # Import all of the compiled Weld blocks called from the transformed function.
                    for module in modules_to_import:
                        augmented_globals[module] = importlib.import_module(module)
                    if eval_cascades is not None:
                        augmented_globals[SMALL_MODEL_NAME] = eval_cascades["small_model"]
                    augmented_globals["scipy"] = importlib.import_module("scipy")
                    augmented_globals["re"] = importlib.import_module("re")
                    augmented_globals["copy"] = importlib.import_module("copy")
                    augmented_globals[WILLUMP_CACHE_NAME] = willump_cache_dict
                    augmented_globals["willump_cache"] = willump_cache
                    augmented_globals["cascade_dense_stacker"] = cascade_dense_stacker
                    augmented_globals["csr_marshall"] = csr_marshall
                    augmented_globals[WILLUMP_TRAINING_CASCADE_NAME] = training_cascades
                    if len(async_funcs) > 0:
                        augmented_globals[WILLUMP_THREAD_POOL_EXECUTOR] = \
                            concurrent.futures.ThreadPoolExecutor(max_workers=5)
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


def execute_from_basics(graph: WillumpGraph, type_map, inputs: tuple, input_names, output_names, aux_data):
    """
    Only for unit tests.  Used to test graph execution separately from inference.
    """
    w_statements = willump.evaluation.willump_weld_generator.graph_to_weld(graph, type_map, None, None, aux_data,
                                                                           1.0, {}, None)
    for entry in w_statements:
        if isinstance(entry, tuple):
            weld_program, _, _ = entry
            break
    weld_program = weld_program[0]
    weld_program = willump.evaluation.willump_weld_generator.set_input_names(weld_program,
                                                                             input_names, aux_data)
    module_name = compile_weld_program(weld_program, type_map=type_map,
                                       input_names=input_names,
                                       output_names=output_names, aux_data=aux_data)
    weld_llvm_caller = importlib.import_module(module_name)
    weld_output, = weld_llvm_caller.caller_func(*inputs)
    return weld_output
