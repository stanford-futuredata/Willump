    return_type* weld_output = (return_type*) weld_output_args->output;

    PyObject* ret_list = PyList_New(0);
    for(int i = 0; i < weld_output->size; i++) {
        i8* str_ptr = weld_output->ptr[i].ptr;
        i64 str_size = weld_output->ptr[i].size;
        PyList_Append(ret_list, PyUnicode_FromStringAndSize((const char *) str_ptr, str_size));
    }

    return ret_list;
}
