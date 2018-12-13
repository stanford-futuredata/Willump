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

    TODO:  Support more input and output types.  Support multiple outputs (maybe).
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
        # Header boilerplate.
        with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller_header.cpp"), "r") as caller_header:
            buffer += caller_header.read()
        # Define the Weld input struct and return type.
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
            typedef %s return_type;
            """ % (input_struct, wtype_to_c_type(output_type))
        # Begin the Weld LLVM caller function.
        buffer += \
            """
            static PyObject *
            caller_func(PyObject *self, PyObject* args) {
            """
        # Define all input variables.
        for i, input_type in enumerate(input_types):
            input_name = "driver_input{0}".format(i)
            if isinstance(input_type, WeldStr):
                buffer += "char* {0} = NULL;\n".format(input_name)
            elif isinstance(input_type, WeldVec):
                if wtype_is_scalar(input_type.elemType):
                    input_array_name = "driver_input_array{0}".format(i)
                    buffer += \
                        """
                        PyObject* {0} = NULL;
                        PyArrayObject* {1} = NULL;
                        """.format(input_name, input_array_name)
                elif isinstance(input_type.elemType, WeldStr):
                    buffer += "PyObject* %s = NULL;\n" % input_name
                else:
                    panic("Unsupported input type {0}".format(str(input_type)))
            elif wtype_is_scalar(input_type):
                buffer += "%s %s;\n" % (str(input_type), input_name)
            else:
                panic("Unsupported input type {0}".format(str(input_type)))
        # Parse all inputs into the input variables.
        format_string = ""
        for input_type in input_types:
            if isinstance(input_type, WeldStr):
                format_string += "s"
            elif isinstance(input_type, WeldVec):
                if wtype_is_scalar(input_type.elemType):
                    format_string += "O!"
                else:
                    format_string += "O"
            elif isinstance(input_type, WeldLong) or isinstance(input_type, WeldInt) or\
                    isinstance(input_type, WeldInt16) or isinstance(input_type, WeldChar):
                format_string += "l"
            elif isinstance(input_type, WeldDouble) or isinstance(input_type, WeldFloat):
                format_string += "d"
        acceptor_string = ""
        for i, input_type in enumerate(input_types):
            input_name = "driver_input{0}".format(i)
            if isinstance(input_type, WeldStr) or wtype_is_scalar(input_type):
                acceptor_string += ", &{0}".format(input_name)
            elif isinstance(input_type, WeldVec):
                if wtype_is_scalar(input_type.elemType):
                    acceptor_string += ", &PyArray_Type, &{0}".format(input_name)
                else:
                    acceptor_string += ", &{0}".format(input_name)
        buffer += \
            """
            if (!PyArg_ParseTuple(args, "%s"%s)) {
                return NULL;
            }
            """ % (format_string, acceptor_string)
        # Convert all input Numpy arrays into PyArrayObjects.
        for i, input_type in enumerate(input_types):
            if isinstance(input_type, WeldVec) and wtype_is_scalar(input_type.elemType):
                input_name = "driver_input{0}".format(i)
                input_array_name = "driver_input_array{0}".format(i)
                buffer += \
                    """
                    if ((%s = (PyArrayObject *) PyArray_FROM_OTF(%s , %s, NPY_ARRAY_IN_ARRAY)) == NULL) {
                        return NULL;
                    }
                    """ % (input_array_name, input_name, weld_type_to_numpy_macro(input_type))
        # Find the length of all vector inputs.
        for i, input_type in enumerate(input_types):
            input_len_name = "input_len%d" % i
            if isinstance(input_type, WeldStr):
                input_name = "driver_input{0}".format(i)
                buffer += "int %s = strlen(%s);\n" % (input_len_name, input_name)
            elif isinstance(input_type, WeldVec) and wtype_is_scalar(input_type.elemType):
                input_array_name = "driver_input_array{0}".format(i)
                buffer += "int %s = PyArray_DIMS(%s)[0];\n" % (input_len_name, input_array_name)
        # Define all the entries of the weld input struct.
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
                if wtype_is_scalar(input_type.elemType):
                    buffer += \
                        """
                        vec<{3}> {0};
                        {0}.size = {1};
                        {0}.ptr = ({3}*) PyArray_DATA({2});
                        """.format(weld_input_name, input_len_name,
                                   input_array_name, wtype_to_c_type(input_type.elemType))
                else:
                    buffer += \
                        """
                        vec<vec<i8>> %s;
                        %s.size = PyList_Size(%s);
                        %s.ptr = (vec<i8>*) malloc(sizeof(vec<i8>) * %s.size);
                        for(int i = 0; i < %s.size; i++) {
                            PyObject* string_entry = PyList_GetItem(%s, i);
                            %s.ptr[i].size = PyUnicode_GET_LENGTH(string_entry);
                            %s.ptr[i].ptr = (i8*) PyUnicode_DATA(string_entry);
                        }
                        """ % (weld_input_name, weld_input_name, input_name, weld_input_name, weld_input_name,
                               weld_input_name, input_name, weld_input_name, weld_input_name)
            elif wtype_is_scalar(input_type):
                buffer += "%s %s = %s;\n" % (wtype_to_c_type(input_type), weld_input_name, input_name)
            buffer += "weld_input._%d = %s;\n" % (i, weld_input_name)
        # Also make inputs out of the aux_data pointers so Weld knows where the data structures are.
        for (aux_i, (input_pointer, input_type)) in enumerate(aux_data):
            i = aux_i + len(input_types)
            weld_input_name = "weld_input%d" % i
            if isinstance(input_type, WeldStr):
                pass
            elif isinstance(input_type, WeldVec):
                buffer += \
                    """
                    {0}* {1} = ({0}*) {3};
                    weld_input._{2}.size = {1}->size;
                    weld_input._{2}.ptr = {1}->ptr;
                    """.format(wtype_to_c_type(input_type), weld_input_name, i, hex(input_pointer))
            elif isinstance(input_type, WeldDict):
                buffer += "weld_input._%d = (void*) %s;\n" % (i, hex(input_pointer))
        # Create the input arguments and run Weld.
        buffer += \
            """    
            struct WeldInputArgs weld_input_args;
            weld_input_args.input = &weld_input;
            weld_input_args.nworkers = 1;
            weld_input_args.memlimit = 100000000;
            weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);
        
            WeldOutputArgs* weld_output_args = run(&weld_input_args);
            return_type* weld_output = (return_type*) weld_output_args->output;
            """
        # Parse Weld outputs and return them.
        if isinstance(output_type, WeldVec) and isinstance(output_type.elemType, WeldStr):
            buffer += \
                """
                PyObject* ret = PyList_New(0);
                for(int i = 0; i < weld_output->size; i++) {
                    i8* str_ptr = weld_output->ptr[i].ptr;
                    i64 str_size = weld_output->ptr[i].size;
                    PyList_Append(ret, PyUnicode_FromStringAndSize((const char *) str_ptr, str_size));
                }
                """
        elif isinstance(output_type, WeldVec) and not isinstance(output_type.elemType, WeldStr):
            buffer += \
                """
                PyArrayObject* ret = 
                    (PyArrayObject*) PyArray_SimpleNewFromData(1, &weld_output->size, %s, weld_output->ptr);
                PyArray_ENABLEFLAGS(ret, NPY_ARRAY_OWNDATA);
                """ % weld_type_to_numpy_macro(output_type)
        else:
            panic("Unrecognized output type %s" % str(output_type))
        buffer += \
            """
                return (PyObject*) ret;
            }
            """
        # Footer boilerplate.
        with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller_footer.cpp"), "r") as footer:
            buffer += footer.read()
        new_function_name = "weld_llvm_caller{0}".format(file_version)
        buffer = buffer.replace("weld_llvm_caller", new_function_name)

    new_file_name = os.path.join(willump_home, "build",
                                 "{0}{1}.cpp".format(base_filename, file_version))
    with open(new_file_name, "w") as outfile:
        outfile.write(buffer)
    return new_file_name


def wtype_is_scalar(wtype: WeldType) -> bool:
    if isinstance(wtype, WeldLong) or isinstance(wtype, WeldInt) or isinstance(wtype, WeldInt16) or \
            isinstance(wtype, WeldChar) or isinstance(wtype, WeldDouble) or isinstance(wtype, WeldFloat):
        return True
    else:
        return False


def wtype_to_c_type(wtype: WeldType) -> str:
    """
    Return the C type used to represent a Weld type in the driver.
    """
    if isinstance(wtype, WeldVec) or isinstance(wtype, WeldStr):
        return "vec<{0}>".format(wtype_to_c_type(wtype.elemType))
    elif isinstance(wtype, WeldDict):
        return "void*"
    else:
        return str(wtype)


def weld_type_to_numpy_macro(wtype: WeldType) -> str:
    """
    Convert a Weld type into a string to plug into the C++ driver.  Currently assumes all types
    are vectors and returns the type of the elements.

    TODO:  More types.
    """
    if isinstance(wtype, WeldVec):
        if isinstance(wtype.elemType, WeldDouble):
            return "NPY_FLOAT64"
        elif isinstance(wtype.elemType, WeldInt):
            return "NPY_INT32"
        elif isinstance(wtype.elemType, WeldLong):
            return "NPY_INT64"
        else:
            panic("Unrecognized IO type {0}".format(wtype.__str__()))
            return ""
    elif isinstance(wtype, WeldStr):
        return "NPY_INT8"
    else:
        panic("Numpy array type that is not vector {0}".format(wtype.__str__()))
        return ""
