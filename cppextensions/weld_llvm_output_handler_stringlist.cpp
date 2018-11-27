    return_type* weld_output = (return_type*) weld_output_args->output;

    PyObject* ret_list = PyList_New(0);
    for(int i = 0; i < weld_output->size; i++) {
        i8* str_ptr = weld_output->ptr[i].ptr;
        PyList_Append(ret_list, PyUnicode_FromString((const char *) str_ptr));
    }

    return ret_list;
}
