import os

from weld.types import *
from willump import panic

from typing import Mapping
import willump.willump_utilities


def generate_cpp_driver(file_version: int, type_map: Mapping[str, WeldType]) -> str:
    """
    Generate versioned CPP files for each run of Willump.

    TODO:  Do C++ code generation to allow multiple inputs and non-vector inputs and outputs.
    """
    willump_home: str = os.environ["WILLUMP_HOME"]
    input_type: WeldType = type_map["__willump_arg0"]
    output_type: WeldType = type_map["__willump_retval"]
    caller_header = open(os.path.join(willump_home, "cppextensions",
                                      "weld_llvm_caller_header.cpp"), "r")
    if isinstance(input_type, WeldStr):
        caller_input_handler = open(os.path.join(willump_home, "cppextensions",
                                             "weld_llvm_input_handler_string.cpp"), "r")
    else:
        caller_input_handler = open(os.path.join(willump_home, "cppextensions",
                                             "weld_llvm_input_handler_numpy.cpp"), "r")
    if isinstance(output_type, WeldVec) and isinstance(output_type.elemType, WeldStr):
        caller_output_handler = open(os.path.join(willump_home, "cppextensions",
                                                 "weld_llvm_output_handler_stringlist.cpp"), "r")
    else:
        caller_output_handler = open(os.path.join(willump_home, "cppextensions",
                                                 "weld_llvm_output_handler_numpy.cpp"), "r")
    caller_footer = open(os.path.join(willump_home, "cppextensions",
                                      "weld_llvm_caller_footer.cpp"), "r")
    buffer: str = caller_header.read() + caller_input_handler.read() + \
        caller_output_handler.read() + caller_footer.read()
    buffer = buffer.replace("WELD_INPUT_TYPE_0", weld_vector_elem_type_to_str(input_type))
    buffer = buffer.replace("WELD_OUTPUT_TYPE", weld_vector_elem_type_to_str(output_type))
    buffer = buffer.replace("NUMPY_INPUT_TYPE_0", weld_type_to_numpy_macro(input_type))
    if "NUMPY_OUTPUT_TYPE" in buffer:
        buffer = buffer.replace("NUMPY_OUTPUT_TYPE", weld_type_to_numpy_macro(output_type))
    new_function_name = "weld_llvm_caller{0}".format(file_version)
    buffer = buffer.replace("weld_llvm_caller", new_function_name)
    new_file_name = os.path.join(willump_home, "build",
                                 "weld_llvm_caller{0}.cpp".format(file_version))
    with open(new_file_name, "w") as outfile:
        outfile.write(buffer)
    caller_header.close()
    caller_input_handler.close()
    caller_output_handler.close()
    caller_footer.close()
    return new_file_name


def weld_vector_elem_type_to_str(wtype: WeldType) -> str:
    """
    Return the string type for the element type of a Weld vector.

    TODO:  More types, not everything is a vector.
    """
    if isinstance(wtype, WeldVec):
        if isinstance(wtype.elemType, WeldVec):
            return "vec<{0}>".format(weld_vector_elem_type_to_str(wtype.elemType))
        elif isinstance(wtype.elemType, WeldStr):
            return "vec<i8>"
        else:
            return str(wtype.elemType)
    elif isinstance(wtype, WeldStr):
        return "i8"
    else:
        panic("Unrecognized IO type {0}".format(wtype.__str__))
        return ""


def weld_type_to_numpy_macro(wtype: WeldType) -> str:
    """
    Convert a Weld type into a string to plug into the C++ driver.  Currently assumes all types
    are vectors and returns the type of the elements.

    TODO:  More types, not everything is a vector.
    """
    if isinstance(wtype, WeldVec):
        if isinstance(wtype.elemType, WeldDouble):
            return "NPY_FLOAT64"
        elif isinstance(wtype.elemType, WeldInt):
            return "NPY_INT32"
        else:
            panic("Unrecognized IO type {0}".format(wtype.__str__))
            return ""
    elif isinstance(wtype, WeldStr):
        return "NPY_INT8"
    else:
        panic("Unrecognized IO type {0}".format(wtype.__str__))
        return ""

