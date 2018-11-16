from weld.types import *
from willump import panic


def weld_scalar_type_to_str(weld_type: WeldType) -> str:
    """
    Converts a Weld scalar type into a string.  Converts a Weld Vector type into the scalar
    type of its elements.
    """
    if isinstance(weld_type, WeldChar):
        return "i8"
    elif isinstance(weld_type, WeldInt16):
        return "i16"
    elif isinstance(weld_type, WeldInt):
        return "i32"
    elif isinstance(weld_type, WeldLong):
        return "i64"
    elif isinstance(weld_type, WeldFloat):
        return "f32"
    elif isinstance(weld_type, WeldDouble):
        return "f64"
    elif isinstance(weld_type, WeldVec):
        return weld_scalar_type_to_str(weld_type.elemType)
    else:
        panic("Invalid Weld type {0}".format(weld_type.__str__))
        return ""