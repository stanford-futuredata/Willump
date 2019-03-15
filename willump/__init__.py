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
WILLUMP_TREES_FEATURE_IMPORTANCES = "willump_trees_feature_importances"
WILLUMP_LINEAR_MODEL_IS_REGRESSION = "willump_linear_model_is_regression"
WILLUMP_KERAS_MODEL_CONFIG = "willump_keras_model_config"
WILLUMP_KERAS_MODEL_WEIGHTS = "willump_keras_model_weights"

# Names
WILLUMP_THREAD_POOL_EXECUTOR = "__willump_threadpool_executor"
WILLUMP_TRAINING_CASCADE_NAME = "__willump_training_cascade"
SMALL_MODEL_NAME = "__willump_small_model"
BIG_MODEL_NAME = "__willump_big_model"
WILLUMP_CACHE_NAME = "__willump_cache"
WILLUMP_CACHE_MAX_LEN_NAME = "__willump_cache_len"
WILLUMP_CACHE_ITER_NUMBER = "__willump_cache_iter_number"


def panic(error: str):
    print("Error: {0}".format(error))
    assert False


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
