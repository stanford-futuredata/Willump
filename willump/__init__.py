import sys
import re
import os

# Static variable names
WILLUMP_LINEAR_REGRESSION_WEIGHTS = "willump_linear_regression_weights"
WILLUMP_LINEAR_REGRESSION_INTERCEPT = "willump_linear_regression_intercept"
WILLUMP_FREQUENCY_COUNT_VOCAB = "willump_frequency_count_vocab"
WILLUMP_COUNT_VECTORIZER_VOCAB = "willump_cv_vocab"
WILLUMP_COUNT_VECTORIZER_ANALYZER = "willump_cv_analyzer"
WILLUMP_COUNT_VECTORIZER_LOWERCASE = "willump_cv_lowercase"
WILLUMP_COUNT_VECTORIZER_NGRAM_RANGE = "willump_cv_ngram_range"
WILLUMP_TFIDF_IDF_VECTOR = "willump_tfidf_idf_vector"
WILLUMP_JOIN_RIGHT_DATAFRAME = "willump_join_right_dataframe"
WILLUMP_JOIN_LEFT_DTYPES = "willump_join_left_dtypes"
WILLUMP_JOIN_HOW = "willump_join_how"
WILLUMP_JOIN_COL = "willump_join_col"
WILLUMP_SUBSCRIPT_INDEX_NAME = "willump_subscript_index"

# Names
WILLUMP_THREAD_POOL_EXECUTOR = "__willump_threadpool_executor"


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
