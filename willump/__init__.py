import sys
import re
import os


def panic(error: str):
    print("Error: {0}".format(error))
    sys.exit(1)


def pprint_weld(weld_prog: str) -> None:
    """
    Pretty-print a Weld program for manual inspection.
    """
    print(re.sub(";", ";\n", weld_prog))


# Ensure Python can find the shared object files Willump will generate.
if "WILLUMP_HOME" not in os.environ:
    print("Error:  WILLUMP_HOME environment variable not defined.  Exiting...")
    sys.exit(0)
willump_build_dir: str = os.path.join(os.environ["WILLUMP_HOME"], "build")
if not os.path.exists(willump_build_dir):
    os.mkdir(willump_build_dir)
sys.path.append(willump_build_dir)
