static PyObject *
caller_func(PyObject *self, PyObject* args)
{
    PyObject *input = NULL;
    PyArrayObject *input_array = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input)) {
        return NULL;
    }
    if((input_array = (PyArrayObject*) PyArray_FROM_OTF(input, NUMPY_INPUT_TYPE_0, NPY_ARRAY_IN_ARRAY)) == NULL) {
        return NULL;
    }
    WELD_INPUT_TYPE_0* data = (WELD_INPUT_TYPE_0 *) PyArray_DATA(input_array);
    int input_len = PyArray_DIMS(input_array)[0];

    vec<WELD_INPUT_TYPE_0> weld_input_vec;
    weld_input_vec.size = input_len;
    weld_input_vec.ptr = data;
    input_type weld_input;
    weld_input._0 = weld_input_vec;

    struct WeldInputArgs weld_input_args;
    weld_input_args.input = &weld_input;
    weld_input_args.nworkers = 1;
    weld_input_args.memlimit = 100000000;
    weld_input_args.run_id = weld_runst_init(weld_input_args.nworkers, weld_input_args.memlimit);

    WeldOutputArgs* weld_output_args = run(&weld_input_args);
