from weld.types import *
import numpy
from willump import panic


def weld_scalar_type_fp(weld_type_str: str = None, weld_type: WeldType = None) -> bool:
    """
    Is a Weld scalar type a floating-point type?
    """
    if weld_type_str is not None:
        if weld_type_str == "f32" or weld_type_str == "f64":
            return True
    else:
        if isinstance(weld_type, WeldFloat) or isinstance(weld_type, WeldDouble):
            return True
    return False


def numpy_type_to_weld_type(numpy_array_dtype) -> WeldType:
    if numpy_array_dtype == numpy.int8:
        return WeldChar()
    elif numpy_array_dtype == numpy.int16:
        return WeldInt16()
    elif numpy_array_dtype == numpy.int32:
        return WeldInt()
    elif numpy_array_dtype == numpy.int64:
        return WeldLong()
    elif numpy_array_dtype == numpy.float16:
        return WeldFloat()
    elif numpy_array_dtype == numpy.float32:
        return WeldFloat()
    elif numpy_array_dtype == numpy.float64:
        return WeldDouble()
    # TODO:  Fix this placeholder.
    elif numpy_array_dtype == numpy.object:
        return WeldStr()
    else:
        panic("Unrecognized Numpy Type %s" % numpy_array_dtype)
        return WeldType()


def strip_linenos_from_var(var_name):
    return var_name[:var_name.rfind("_")]
