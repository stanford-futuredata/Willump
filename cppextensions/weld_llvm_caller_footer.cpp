static PyMethodDef CallerMethods[] = {
 { "caller_func", caller_func, METH_VARARGS, "Call Weld LLVM." },
 { NULL, NULL, 0, NULL }
};

static struct PyModuleDef weld_llvm_caller_module = {
    PyModuleDef_HEAD_INIT,
    "weld_llvm_caller",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CallerMethods
};

PyMODINIT_FUNC
PyInit_weld_llvm_caller(void)
{
    import_array();
    return PyModule_Create(&weld_llvm_caller_module);
}
