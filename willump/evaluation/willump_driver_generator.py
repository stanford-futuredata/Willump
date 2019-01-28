import os

from weld.types import *
from willump import panic
from willump.willump_utilities import weld_scalar_type_fp

from typing import Mapping, List, Tuple


def generate_cpp_driver(file_version: int, type_map: Mapping[str, WeldType],
                        input_names: List[str], output_names: List[List[str]],
                        base_filename: str, aux_data: List[Tuple[int, WeldType]],
                        thread_runner_pointer: int, entry_point_names: List[str]) -> str:
    """
    Generate a versioned CPP driver for a Weld program. If base_filename is not
    weld_llvm_caller, assume the driver already exists at
    WILLUMP_HOME/cppextensions/base_filename.cpp.  Otherwise, generate a driver using the
    Weld inputs and outputs in type_map.
    """
    willump_home: str = os.environ["WILLUMP_HOME"]
    if base_filename is not "weld_llvm_caller":
        assert(len(entry_point_names) == 1)
        if base_filename is "hash_join_dataframe_indexer":
            buffer = generate_hash_join_dataframe_indexer_driver(type_map, input_names)
        else:
            with open(os.path.join(willump_home, "cppextensions", base_filename + ".cpp")) as driver:
                buffer = driver.read()
        buffer = buffer.replace(base_filename, base_filename + str(file_version))
        buffer = buffer.replace("WELD_ENTRY_POINT", entry_point_names[0])
    else:
        def name_typer(name): return type_map[name]
        input_types: List[WeldType] = list(map(name_typer, input_names))
        output_types_list: List[List[WeldType]] = list(map(lambda type_list: list(map(name_typer, type_list)), output_names))
        num_outputs = sum(map(len, output_names))
        buffer = ""
        # Header boilerplate.
        with open(os.path.join(willump_home, "cppextensions", "weld_llvm_caller_header.cpp"), "r") as caller_header:
            buffer += caller_header.read()
        buffer += "weld_thread_runner* thread_runner = (weld_thread_runner*) %s;" % str(hex(thread_runner_pointer))
        for weld_entry_point in entry_point_names:
            buffer += "extern \"C\" struct WeldOutputArgs* %s(struct WeldInputArgs*);" % weld_entry_point
        # Define the Weld input struct and output struct.
        input_struct = ""
        for i, input_type in enumerate(input_types):
            if isinstance(input_type, WeldPandas):
                inner_struct = ""
                for inner_i, inner_type in enumerate(input_type.field_types):
                    inner_struct += "{0} _{1};\n".format(wtype_to_c_type(inner_type), inner_i)
                buffer += \
                    """
                    struct struct_in_%d {
                      %s
                    };
                    """ % (i, inner_struct)
                input_struct += "struct struct_in_{0} _{1};\n".format(i, i)
            else:
                input_struct += "{0} _{1};\n".format(wtype_to_c_type(input_type), i)
        for (i, (_, input_type)) in enumerate(aux_data):
            input_struct += "{0} _{1};\n".format(wtype_to_c_type(input_type), i + len(input_types))
        buffer += \
            """
            struct struct_in {
              %s
            };
            typedef struct_in input_type;
            """ % input_struct
        for output_num, output_types in enumerate(output_types_list):
            output_struct = ""
            for i, output_type in enumerate(output_types):
                if isinstance(output_type, WeldPandas) or isinstance(output_type, WeldCSR):
                    inner_struct = ""
                    for inner_i, inner_type in enumerate(output_type.field_types):
                        inner_struct += "{0} _{1};\n".format(wtype_to_c_type(inner_type), inner_i)
                    buffer += \
                        """
                        struct weld_struct_%d_%d {
                          %s
                        };
                        """ % (output_num, i, inner_struct)
                    output_struct += "struct weld_struct_%d_%d _%d;\n" % (output_num, i, i)
                else:
                    output_struct += "{0} _{1};\n".format(wtype_to_c_type(output_type), i)
            buffer += \
                """
                struct struct_out_%d {
                  %s
                };
                typedef struct_out_%d return_type_%d;
                """ % (output_num, output_struct, output_num, output_num)
        # Begin the Weld LLVM caller function.
        buffer += \
            """
            static PyObject *
            caller_func(PyObject *self, PyObject* args) {
            """
        # Generate the input parser
        buffer += generate_input_parser(input_types, aux_data)
        # Create the input arguments and run Weld.
        buffer += \
            """    
            struct WeldInputArgs weld_input_args;
            weld_input_args.input = &weld_input;
            weld_input_args.nworkers = 1;
            weld_input_args.memlimit = 10000000000;
            weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);
            
            /*thread_runner->run_function = WELD_ENTRY_POINT;
            thread_runner->argument = &weld_input_args;
            thread_runner->done = false;
            thread_runner->ready = true;
            //printf("Driver Thread CPU ID %d\\n", sched_getcpu());
            while(1) {
                atomic_thread_fence(memory_order_acquire);
                if(thread_runner->done) {
                    break;
                }
            }
            WeldOutputArgs* weld_output_args = thread_runner->output;*/
            WeldOutputArgs* weld_output_args;
            """
        for output_num, weld_entry_point in enumerate(entry_point_names):
            buffer += \
                """
                weld_output_args = %s(&weld_input_args);
                return_type_%d* weld_output_%d = (return_type_%d*) weld_output_args->output;
                """ % (weld_entry_point, output_num, output_num, output_num)
        # Marshall the output into ret_tuple ordered as in output_names.
        buffer += \
            """
            PyObject* ret_tuple = PyTuple_New(%d);
            """ % num_outputs
        for output_num, output_types in enumerate(output_types_list):
            buffer += generate_output_parser(output_num, output_types)
        buffer += \
            """
                return (PyObject*) ret_tuple;
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


def generate_input_parser(input_types: List[WeldType], aux_data) -> str:
    # Define all input variables.
    buffer = ""
    for i, input_type in enumerate(input_types):
        input_name = "driver_input{0}".format(i)
        if isinstance(input_type, WeldStr):
            buffer += "char* {0} = NULL;\n".format(input_name)
        elif isinstance(input_type, WeldVec):
            if wtype_is_scalar(input_type.elemType) or isinstance(input_type.elemType, WeldVec):
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
        elif isinstance(input_type, WeldPandas):
            buffer += "PyObject* {0} = NULL;\n".format(input_name)
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
            if not isinstance(input_type.elemType, WeldStr):
                format_string += "O!"
            else:
                format_string += "O"
        elif isinstance(input_type, WeldPandas):
            format_string += "O"
        elif isinstance(input_type, WeldLong) or isinstance(input_type, WeldInt) or \
                isinstance(input_type, WeldInt16) or isinstance(input_type, WeldChar):
            format_string += "l"
        elif isinstance(input_type, WeldDouble) or isinstance(input_type, WeldFloat):
            format_string += "d"
    acceptor_string = ""
    for i, input_type in enumerate(input_types):
        input_name = "driver_input{0}".format(i)
        if isinstance(input_type, WeldStr) or wtype_is_scalar(input_type) or isinstance(input_type, WeldPandas):
            acceptor_string += ", &{0}".format(input_name)
        elif isinstance(input_type, WeldVec):
            if not isinstance(input_type.elemType, WeldStr):
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
        if isinstance(input_type, WeldVec) and not isinstance(input_type.elemType, WeldStr):
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
        elif isinstance(input_type, WeldVec) and not isinstance(input_type.elemType, WeldStr):
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
            elif isinstance(input_type.elemType, WeldStr):
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
            elif isinstance(input_type.elemType, WeldVec):
                buffer += \
                    """
                    vec<vec<{0}>> {1};
                    {1}.size = {2};
                    {1}.ptr = (vec<{0}>*) malloc(sizeof(vec<{0}>) * {2});
                    int ele_size{3} = PyArray_DIMS({4})[1];
                    for(int i = 0; i < {2}; i++) {{
                        {1}.ptr[i].size = ele_size{3};
                        {1}.ptr[i].ptr = (({0}*) PyArray_DATA({4})) + i * ele_size{3};
                    }}
                    """.format(wtype_to_c_type(input_type.elemType.elemType),
                               weld_input_name, input_len_name, i, input_array_name)
            else:
                panic("Unrecognized elemType in input WeldVec %s" % str(input_type.elemType))
        elif isinstance(input_type, WeldPandas):
            buffer += \
                """
                struct struct_in_%d %s;
                """ % (i, weld_input_name)
            for inner_i, field_type in enumerate(input_type.field_types):
                buffer += \
                    """
                    PyObject* weld_entry{0} = PyTuple_GetItem({1}, {0});
                    """.format(inner_i, input_name)
                if isinstance(field_type, WeldVec):
                    buffer += \
                        """
                        PyArrayObject* weld_numpy_entry%d;
                        if ((weld_numpy_entry%d = (PyArrayObject *) PyArray_FROM_OTF(weld_entry%d , %s, 
                            NPY_ARRAY_IN_ARRAY)) == NULL) {
                            return NULL;
                        }
                        """ % (inner_i, inner_i, inner_i, weld_type_to_numpy_macro(field_type))
                    buffer += \
                        """
                        {0}._{1}.size = PyArray_DIMS(weld_numpy_entry{1})[0];
                        {0}._{1}.ptr = ({2}*) PyArray_DATA(weld_numpy_entry{1});
                        """.format(weld_input_name,
                                   inner_i, wtype_to_c_type(field_type.elemType))
                elif wtype_is_scalar(field_type):
                    if weld_scalar_type_fp(weld_type=field_type):
                        buffer += \
                            """
                            %s._%d = PyFloat_AS_DOUBLE(weld_entry%d);
                            """ % (weld_input_name, inner_i, inner_i)
                    else:
                        buffer += \
                            """
                            %s._%d = PyLong_AsLong(weld_entry%d);
                            """ % (weld_input_name, inner_i, inner_i)
                else:
                    panic("Unrecognized struct field type %s" % str(field_type))
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
    return buffer


def generate_output_parser(output_num: int, output_types: List[WeldType]) -> str:
    buffer = ""
    # Parse Weld outputs and return them.
    for i, output_type in enumerate(output_types):
        buffer += \
            """
            {
            """
        if isinstance(output_type, WeldPandas) or isinstance(output_type, WeldCSR):
            buffer += "struct weld_struct_%d_%d curr_output = weld_output_%d->_%d;\n" % (output_num, i, output_num, i)
        else:
            buffer += "%s curr_output = weld_output_%d->_%d;\n" % (wtype_to_c_type(output_type), output_num, i)
        if isinstance(output_type, WeldVec) and isinstance(output_type.elemType, WeldStr):
            buffer += \
                """
                PyObject* ret = PyList_New(0);
                for(int i = 0; i < curr_output.size; i++) {
                    i8* str_ptr = curr_output.ptr[i].ptr;
                    i64 str_size = curr_output.ptr[i].size;
                    PyList_Append(ret, PyUnicode_FromStringAndSize((const char *) str_ptr, str_size));
                }
                """
        # TODO:  Return a 2-D array instead of a list of 1-D arrays.
        elif isinstance(output_type, WeldVec) and isinstance(output_type.elemType, WeldVec):
            buffer += \
                """
                PyObject* ret = PyList_New(0);
                for(int i = 0; i < curr_output.size; i++) {
                    %s* entry_ptr = curr_output.ptr[i].ptr;
                    i64 entry_size = curr_output.ptr[i].size;
                    PyArrayObject* ret_entry = 
                        (PyArrayObject*) PyArray_SimpleNewFromData(1, &entry_size, %s, entry_ptr);
                    PyArray_ENABLEFLAGS(ret_entry, NPY_ARRAY_OWNDATA);
                    PyList_Append(ret, (PyObject*) ret_entry);
                }
                """ % (str(output_type.elemType.elemType), weld_type_to_numpy_macro(output_type.elemType))
        elif isinstance(output_type, WeldVec):
            buffer += \
                """
                PyArrayObject* ret = 
                    (PyArrayObject*) PyArray_SimpleNewFromData(1, &curr_output.size, %s, curr_output.ptr);
                PyArray_ENABLEFLAGS(ret, NPY_ARRAY_OWNDATA);
                """ % weld_type_to_numpy_macro(output_type)
        elif isinstance(output_type, WeldPandas) or isinstance(output_type, WeldCSR):
            field_types = output_type.field_types
            buffer += \
                """
                PyObject *ret = PyTuple_New(%d);
                PyObject* ret_entry;
                """ % len(field_types)
            for inner_i, field_type in enumerate(field_types):
                if isinstance(field_type, WeldVec):
                    buffer += \
                        """
                        ret_entry = (PyObject*) 
                            PyArray_SimpleNewFromData(1, &curr_output._%d.size, %s, curr_output._%d.ptr);
                        //PyArray_ENABLEFLAGS((PyArrayObject*) ret_entry, NPY_ARRAY_OWNDATA);
                        PyTuple_SetItem(ret, %d, ret_entry);
                        """ % (inner_i, weld_type_to_numpy_macro(field_type), inner_i, inner_i)
                elif wtype_is_scalar(field_type):
                    if weld_scalar_type_fp(weld_type=field_type):
                        buffer += \
                            """
                            ret_entry =
                                PyFloat_FromDouble(curr_output._%d);
                            PyTuple_SetItem(ret, %d, ret_entry);
                            """ % (inner_i, inner_i)
                    else:
                        buffer += \
                            """
                            ret_entry =
                                PyLong_FromLong((long) curr_output._%d);
                            PyTuple_SetItem(ret, %d, ret_entry);
                            """ % (inner_i, inner_i)
                else:
                    panic("Unrecognized struct field type %s" % str(field_type))
        else:
            panic("Unrecognized output type %s" % str(output_type))
        buffer += \
            """
            PyTuple_SetItem(ret_tuple, %d, (PyObject*) ret);
            }
            """ % (output_num + i)
    return buffer


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
    if isinstance(wtype, WeldVec) or isinstance(wtype, WeldCSR):
        if isinstance(wtype.elemType, WeldDouble):
            return "NPY_FLOAT64"
        elif isinstance(wtype.elemType, WeldFloat):
            return "NPY_FLOAT32"
        elif isinstance(wtype.elemType, WeldChar):
            return "NPY_INT8"
        elif isinstance(wtype.elemType, WeldInt16):
            return "NPY_INT16"
        elif isinstance(wtype.elemType, WeldInt):
            return "NPY_INT32"
        elif isinstance(wtype.elemType, WeldLong):
            return "NPY_INT64"
        elif isinstance(wtype, WeldVec):
            return weld_type_to_numpy_macro(wtype.elemType)
        else:
            panic("Unrecognized IO type {0}".format(wtype.__str__()))
            return ""
    elif isinstance(wtype, WeldStr):
        return "NPY_INT8"
    else:
        panic("Numpy array type that is not vector {0}".format(wtype.__str__()))
        return ""


def generate_hash_join_dataframe_indexer_driver(type_map: Mapping[str, WeldType], input_names: List[str]) -> str:
    willump_home: str = os.environ["WILLUMP_HOME"]
    with open(os.path.join(willump_home, "cppextensions", "hash_join_dataframe_indexer.cpp")) as driver:
        buffer = driver.read()
    input_types: List[WeldType] = list(map(lambda x: type_map[x], input_names))
    input_struct = ""
    for i, input_type in enumerate(input_types):
        input_struct += "{0} _{1};\n".format(wtype_to_c_type(input_type), i)
    buffer = buffer.replace("INPUT_STRUCT_CONTENTS", input_struct)
    buffer = buffer.replace("INPUT_PARSING_CONTENTS", generate_input_parser(input_types, []))
    return buffer
