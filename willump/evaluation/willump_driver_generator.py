import os

from weld.types import *
from willump import panic

from typing import Mapping, List, Tuple


def generate_cpp_driver(file_version: int, type_map: Mapping[str, WeldType],
                        base_filename: str, aux_data: List[Tuple[int, WeldType]]) -> str:
    """
    Generate a versioned CPP driver for a Weld program. If base_filename is not
    weld_llvm_caller, assume the driver already exists at
    WILLUMP_HOME/cppextensions/base_filename.cpp.  Otherwise, generate a driver using the
    Weld inputs and outputs in type_map.

    TODO:  Do C++ generation properly for arbitrary combinations of inputs and auxiliary data.
    """
    willump_home: str = os.environ["WILLUMP_HOME"]
    if base_filename is not "weld_llvm_caller":
        with open(os.path.join(willump_home, "cppextensions", base_filename + ".cpp")) as driver:
            buffer = driver.read()
            buffer = buffer.replace(base_filename, base_filename + str(file_version))
    else:
        input_types: List[WeldType] = []
        num_inputs = 0
        while "__willump_arg{0}".format(num_inputs) in type_map:
            input_types.append(type_map["__willump_arg{0}".format(num_inputs)])
            num_inputs += 1
        output_type: WeldType = type_map["__willump_retval"]
        buffer = ""
        with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller_header.cpp"), "r") as caller_header:
            buffer += caller_header.read()
        input_struct = ""
        for i, input_type in enumerate(input_types):
            input_struct += "{0} _{1};\n".format(wtype_to_c_type(input_type), i)
        for (i, (_, input_type)) in enumerate(aux_data):
            input_struct += "{0} _{1};\n".format(wtype_to_c_type(input_type), i + len(input_types))
        buffer += \
            """
            struct struct0 {
              %s
            };
            typedef struct0 input_type;
            """ % input_struct

        if len(aux_data) == 0:
            buffer += \
                """
                static PyObject *
                caller_func(PyObject *self, PyObject* args) {
                """
            for i, input_type in enumerate(input_types):
                input_name = "driver_input{0}".format(i)
                if isinstance(input_type, WeldStr):
                    buffer += "char* {0} = NULL;\n".format(input_name)
                elif isinstance(input_type, WeldVec):
                    input_array_name = "driver_input_array{0}".format(i)
                    buffer += \
                        """
                        PyObject* {0} = NULL;
                        PyArrayObject* {1} = NULL;
                        """.format(input_name, input_array_name)
                else:
                    panic("Unsupported input type {0}".format(str(input_type)))
            format_string = ""
            for input_type in input_types:
                if isinstance(input_type, WeldStr):
                    format_string += "s"
                elif isinstance(input_type, WeldVec):
                    format_string += "O!"
            acceptor_string = ""
            for i, input_type in enumerate(input_types):
                input_name = "driver_input{0}".format(i)
                if isinstance(input_type, WeldStr):
                    acceptor_string += ", &{0}".format(input_name)
                elif isinstance(input_type, WeldVec):
                    acceptor_string += ", &PyArray_Type, &{0}".format(input_name)
            buffer += \
                """
                if (!PyArg_ParseTuple(args, "%s"%s)) {
                    return NULL;
                }
                """ % (format_string, acceptor_string)
            for i, input_type in enumerate(input_types):
                if isinstance(input_type, WeldVec):
                    input_name = "driver_input{0}".format(i)
                    input_array_name = "driver_input_array{0}".format(i)
                    buffer += \
                        """
                        if ((%s = (PyArrayObject *) PyArray_FROM_OTF(%s , %s, NPY_ARRAY_IN_ARRAY)) == NULL) {
                            return NULL;
                        }
                        """ % (input_array_name, input_name, weld_type_to_numpy_macro(input_type))
            for i, input_type in enumerate(input_types):
                input_len_name = "input_len%d" % i
                if isinstance(input_type, WeldStr):
                    input_name = "driver_input{0}".format(i)
                    buffer += "int %s = strlen(%s);\n" % (input_len_name, input_name)
                elif isinstance(input_type, WeldVec):
                    input_array_name = "driver_input_array{0}".format(i)
                    buffer += "int %s = PyArray_DIMS(%s)[0];\n" % (input_len_name, input_array_name)
            buffer += "input_type weld_input;\n"
            for i, input_type in enumerate(input_types):
                input_len_name = "input_len%d" % i
                input_name = "driver_input{0}".format(i)
                input_array_name = "driver_input_array{0}".format(i)
                weld_input_name = "weld_input%d" % i
                if isinstance(input_type, WeldStr):
                    buffer += \
                        """
                        vec<i8> {0};
                        {0}.size = {1};
                        {0}.ptr = (i8*) {2};
                        """.format(weld_input_name, input_len_name, input_name)
                elif isinstance(input_type, WeldVec):
                    buffer += \
                        """
                        vec<{3}> {0};
                        {0}.size = {1};
                        {0}.ptr = ({3}*) PyArray_DATA({2});
                        """.format(weld_input_name, input_len_name,
                                   input_array_name, wvector_elem_type_to_str(input_type))
                buffer += "weld_input._%d = %s;\n" % (i, weld_input_name)

            buffer += \
                """    
                struct WeldInputArgs weld_input_args;
                weld_input_args.input = &weld_input;
                weld_input_args.nworkers = 1;
                weld_input_args.memlimit = 100000000;
                weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);
            
                WeldOutputArgs* weld_output_args = run(&weld_input_args);
                """
            if isinstance(output_type, WeldVec) and isinstance(output_type.elemType, WeldStr):
                with open(os.path.join(willump_home, "cppextensions",
                            "weld_llvm_output_handler_stringlist.cpp"), "r") as output_handler:
                    buffer += output_handler.read()
            elif isinstance(output_type, WeldVec) and not isinstance(output_type.elemType, WeldStr):
                with open(os.path.join(willump_home, "cppextensions",
                            "weld_llvm_output_handler_numpy.cpp"), "r") as output_handler:
                    buffer += output_handler.read()
            else:
                panic("Unrecognized output type %s" % str(output_type))
            with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller_footer.cpp"), "r") as footer:
                buffer += footer.read()

            buffer = buffer.replace("WELD_OUTPUT_TYPE", wvector_elem_type_to_str(output_type))
            if "NUMPY_OUTPUT_TYPE" in buffer:
                buffer = buffer.replace("NUMPY_OUTPUT_TYPE", weld_type_to_numpy_macro(output_type))
            new_function_name = "weld_llvm_caller{0}".format(file_version)
            buffer = buffer.replace("weld_llvm_caller", new_function_name)
        else:
            if len(aux_data) == 2:
                caller_input_handler = open(os.path.join(willump_home, "cppextensions",
                                                     "weld_llvm_input_handler_string_dict.cpp"), "r")
            elif len(aux_data) > 2:
                caller_input_handler = open(os.path.join(willump_home, "cppextensions",
                                                     "weld_llvm_input_handler_freq_logit.cpp"), "r")
            elif isinstance(input_types[0], WeldStr):
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
            buffer = buffer + caller_input_handler.read() + \
                caller_output_handler.read() + caller_footer.read()
            buffer = buffer.replace("WELD_INPUT_TYPE_0", wvector_elem_type_to_str(input_types[0]))
            buffer = buffer.replace("WELD_OUTPUT_TYPE", wvector_elem_type_to_str(output_type))
            buffer = buffer.replace("NUMPY_INPUT_TYPE_0", weld_type_to_numpy_macro(input_types[0]))
            if "NUMPY_OUTPUT_TYPE" in buffer:
                buffer = buffer.replace("NUMPY_OUTPUT_TYPE", weld_type_to_numpy_macro(output_type))
            for i in range(len(aux_data)):
                buffer = buffer.replace("POINTER{0}".format(i), hex(aux_data[i][0]))
            new_function_name = "weld_llvm_caller{0}".format(file_version)
            buffer = buffer.replace("weld_llvm_caller", new_function_name)
            caller_header.close()
            caller_input_handler.close()
            caller_output_handler.close()
            caller_footer.close()
    new_file_name = os.path.join(willump_home, "build",
                                 "{0}{1}.cpp".format(base_filename, file_version))
    with open(new_file_name, "w") as outfile:
        outfile.write(buffer)
    return new_file_name


def wtype_to_c_type(wtype: WeldType) -> str:
    """
    Return the C type used to represent a Weld type in the driver.
    """
    if isinstance(wtype, WeldVec):
        return "vec<{0}>".format(wtype_to_c_type(wtype.elemType))
    elif isinstance(wtype, WeldStr):
        return "vec<i8>"
    elif isinstance(wtype, WeldDict):
        return "void*"
    else:
        return str(wtype)


def wvector_elem_type_to_str(wtype: WeldType) -> str:
    """
    Return the string type for the element type of a Weld vector.

    TODO:  More types, not everything is a vector.
    """
    if isinstance(wtype, WeldVec):
        if isinstance(wtype.elemType, WeldVec):
            return "vec<{0}>".format(wvector_elem_type_to_str(wtype.elemType))
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
        elif isinstance(wtype.elemType, WeldLong):
            return "NPY_INT64"
        else:
            panic("Unrecognized IO type {0}".format(wtype.__str__))
            return ""
    elif isinstance(wtype, WeldStr):
        return "NPY_INT8"
    else:
        panic("Unrecognized IO type {0}".format(wtype.__str__))
        return ""

