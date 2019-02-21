import unittest
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
import time

import willump.evaluation.willump_executor as wexec


@wexec.willump_execute()
def sample_pandas_unpacked(array_one, array_two):
    df = pd.DataFrame()
    df["a"] = array_one
    df["b"] = array_two
    df_a = df["a"].values
    df_b = df["b"].values
    sum_var = df_a + df_b
    mul_sum_var = sum_var * sum_var
    df["ret"] = mul_sum_var
    return df


@wexec.willump_execute()
def sample_pandas_unpacked_nested_ops(array_one, array_two, array_three):
    df = pd.DataFrame()
    df["a"] = array_one
    df["b"] = array_two
    df["c"] = array_three
    df_a = df["a"].values
    df_b = df["b"].values
    df_c = df["c"].values
    op_var = (df_a - df_c) + df_b * df_c
    df["ret"] = op_var
    return df


with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=0.005, max_df=1.0,
                             lowercase=False, stop_words=None, binary=False, decode_error='replace',
                             vocabulary=simple_vocab_dict)


@wexec.willump_execute()
def sample_pandas_count_vectorizer(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


@wexec.willump_execute()
def sample_pandas_merge(left, right):
    new = left.merge(right, how="left", on="join_column")
    return new


model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)


@wexec.willump_execute()
def sample_logistic_regression_np_array(np_input):
    predicted_result = model.predict(np_input)
    return predicted_result


@wexec.willump_execute()
def sample_logistic_regression_cv(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    predicted_result = model.predict(transformed_result)
    return predicted_result


FEATURES = ["data1", "data2", "metadata1", "metadata2"]


@wexec.willump_execute()
def sample_merge_linear_regression(left, right):
    new = left.merge(right, how="left", on="join_column")
    feature_columns = new[FEATURES]
    predicted_result = model.predict(feature_columns)
    return predicted_result


@wexec.willump_execute()
def sample_merge_linear_regression_nobatch(left_series, right):
    left = left_series.to_frame().T
    new = left.merge(right, how="left", on="join_column")
    feature_columns = new[FEATURES]
    predicted_result = model.predict(feature_columns)
    return predicted_result


DATA_FEATURES = ["data1", "data2"]


@wexec.willump_execute()
def sample_column_select(df):
    data_df = df[DATA_FEATURES]
    return data_df


def first():
    time.sleep(0.1)
    return 1


def second():
    time.sleep(0.1)
    return 2


@wexec.willump_execute(async_funcs=["first", "second"])
def sample_async_funcs(three):
    one = first()
    two = second()
    six = one + two + three
    return six


@wexec.willump_execute()
def pandas_sample_string_lower(df):
    df["new_col"] = df["title"].apply(str.lower)
    return df


@wexec.willump_execute()
def pandas_series_to_df(pandas_series):
    df = pandas_series.to_frame().T
    return df


@wexec.willump_execute()
def pandas_series_concat(series_one, series_two):
    combined_series = pd.concat([series_one, series_two], axis=0)
    combined_series = combined_series.values
    return combined_series


class PandasGraphInferenceTests(unittest.TestCase):
    def test_unpacked_pandas(self):
        print("\ntest_unpacked_pandas")
        vec_one = numpy.array([1., 2., 3.], dtype=numpy.float64)
        vec_two = numpy.array([4., 5., 6.], dtype=numpy.float64)
        sample_pandas_unpacked(vec_one, vec_two)
        sample_pandas_unpacked(vec_one, vec_two)
        weld_output = sample_pandas_unpacked(vec_one, vec_two)
        numpy.testing.assert_almost_equal(weld_output["ret"].values, numpy.array([25., 49., 81.]))

    def test_unpacked_pandas_nested_ops(self):
        print("\ntest_unpacked_pandas_nested_ops")
        vec_one = numpy.array([1., 2., 3.], dtype=numpy.float64)
        vec_two = numpy.array([4., 5., 6.], dtype=numpy.float64)
        vec_three = numpy.array([0., 1., 0.], dtype=numpy.float64)
        sample_pandas_unpacked_nested_ops(vec_one, vec_two, vec_three)
        sample_pandas_unpacked_nested_ops(vec_one, vec_two, vec_three)
        weld_output = sample_pandas_unpacked_nested_ops(vec_one, vec_two, vec_three)
        numpy.testing.assert_almost_equal(weld_output["ret"].values, numpy.array([1., 6., 3.]))

    def test_pandas_count_vectorizer(self):
        print("\ntest_pandas_count_vectorizer")
        string_array = ["theaancatdog house", "bobthe builder", "an    \t   ox anox an ox"]
        correct_output = sample_pandas_count_vectorizer(string_array, vectorizer)
        sample_pandas_count_vectorizer(string_array, vectorizer)
        weld_output = sample_pandas_count_vectorizer(string_array, vectorizer)
        numpy.testing.assert_almost_equal(weld_output.toarray(), correct_output.toarray())

    def test_pandas_join(self):
        print("\ntest_pandas_join")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        sample_pandas_merge(left_table, right_table)
        sample_pandas_merge(left_table, right_table)
        weld_output = sample_pandas_merge(left_table, right_table)
        numpy.testing.assert_equal(
            weld_output["join_column"].values, left_table["join_column"].values)
        numpy.testing.assert_equal(
            weld_output["data1"].values, numpy.array([4, 5, 2, 5, 3], dtype=numpy.int64))
        numpy.testing.assert_equal(
            weld_output["metadata1"].values, numpy.array([1.2, 2.2, 2.2, 3.2, 1.2], dtype=numpy.float64))
        numpy.testing.assert_equal(
            weld_output["metadata2"].values, numpy.array([1.3, 2.3, 2.3, 3.3, 1.3], dtype=numpy.float64))

    def test_pandas_regression_vec(self):
        print("\ntest_pandas_regression_vec")
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        input_vec = numpy.array([[0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 3, 0]], dtype=numpy.int64)
        sample_logistic_regression_np_array(input_vec)
        sample_logistic_regression_np_array(input_vec)
        weld_output = sample_logistic_regression_np_array(input_vec)
        numpy.testing.assert_equal(
            weld_output, numpy.array([1, 0], dtype=numpy.int64))

    def test_logistic_regression_cv(self):
        print("\ntest_logistic_regression_cv")
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        string_array = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dog the the the the"]
        sample_logistic_regression_cv(string_array, vectorizer)
        sample_logistic_regression_cv(string_array, vectorizer)
        weld_output = sample_logistic_regression_cv(string_array, vectorizer)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0, 1, 0, 1], dtype=numpy.int64))

    def test_simple_join_regression_batch(self):
        print("\ntest_simple_join_regression_batch")
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        sample_merge_linear_regression(left_table, right_table)
        sample_merge_linear_regression(left_table, right_table)
        weld_output = sample_merge_linear_regression(left_table, right_table)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0, 1, 1, 1, 1], dtype=numpy.int64))

    def test_simple_join_regression_nobatch(self):
        print("\ntest_simple_join_regression_nobatch")
        model.coef_ = numpy.array([[0.1, 0.2, 0.3, 0.4]], dtype=numpy.float64)
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv").iloc[0]
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        sample_merge_linear_regression_nobatch(left_table, right_table)
        sample_merge_linear_regression_nobatch(left_table, right_table)
        weld_output = sample_merge_linear_regression_nobatch(left_table, right_table)
        numpy.testing.assert_equal(
            weld_output, numpy.array([0], dtype=numpy.int64))

    def test_column_select(self):
        print("\ntest_column_select")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        sample_column_select(left_table)
        sample_column_select(left_table)
        weld_output = sample_column_select(left_table)
        numpy.testing.assert_equal(
            weld_output["data2"].values, left_table["data2"].values)
        numpy.testing.assert_equal(
            weld_output["data1"].values, numpy.array([4, 5, 2, 5, 3], dtype=numpy.int64))
        self.assertEqual(len(weld_output.columns), 2)

    def test_async_funcs(self):
        print("\ntest_async_funcs")
        sample_async_funcs(3)
        sample_async_funcs(3)
        start = time.time()
        weld_output = sample_async_funcs(3)
        end = time.time()
        time_elapsed = end - start
        self.assertLess(time_elapsed, 0.2)
        self.assertEqual(weld_output, 6)

    def test_pandas_string_lower(self):
        print("\ntest_pandas_string_lower")
        input_df = pd.DataFrame({"title": ["aA,.,.a", "B,,b", "c34234C"]})
        pandas_sample_string_lower(input_df)
        pandas_sample_string_lower(input_df)
        weld_output = pandas_sample_string_lower(input_df)
        self.assertEqual(list(weld_output["new_col"].values), ["aa,.,.a", "b,,b", "c34234c"])
        self.assertEqual(list(weld_output["title"].values), ["aA,.,.a", "B,,b", "c34234C"])

    def test_series_to_df(self):
        print("\ntest_series_to_df")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        input_series = left_table.iloc[0]
        correct_output = pandas_series_to_df(input_series)
        pandas_series_to_df(input_series)
        weld_output = pandas_series_to_df(input_series)
        numpy.testing.assert_almost_equal(correct_output.values, weld_output.values)

    def test_series_concat(self):
        print("\ntest_series_concat")
        left_table = pd.read_csv("tests/test_resources/toy_data_csv.csv")
        right_table = pd.read_csv("tests/test_resources/toy_metadata_csv.csv")
        right_table = right_table.drop("join_column", axis=1)
        series_one = left_table.iloc[0]
        series_two = right_table.iloc[0]
        correct_output = pandas_series_concat(series_one, series_two)
        pandas_series_concat(series_one, series_two)
        weld_output = pandas_series_concat(series_one, series_two)
        numpy.testing.assert_almost_equal(correct_output, weld_output)
