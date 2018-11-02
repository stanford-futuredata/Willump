import sys
import re


def panic(error: str):
    print("Error: {0}".format(str))
    sys.exit(1)


def pprint_weld(weld_prog: str) -> None:
    """
    Pretty-print a Weld program for manual inspection.
    """
    print(re.sub(";", ";\n", weld_prog))