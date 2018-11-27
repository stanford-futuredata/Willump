    return_type* weld_output = (return_type*) weld_output_args->output;

    PyArrayObject* return_array = (PyArrayObject*) PyArray_SimpleNewFromData(1, &weld_output->size, NUMPY_OUTPUT_TYPE, weld_output->ptr);
    PyArray_ENABLEFLAGS(return_array, NPY_ARRAY_OWNDATA);

    return (PyObject*) return_array;
}
