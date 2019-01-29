import unittest
import numpy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import willump.evaluation.willump_executor as wexec
import scipy.sparse.csr
import scipy.sparse

with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=0.005, max_df=1.0,
                             lowercase=False, stop_words=None, binary=False, decode_error='replace',
                             vocabulary=simple_vocab_dict)


@wexec.willump_execute()
def sample_stack_sparse(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    transformed_result = scipy.sparse.hstack([transformed_result, transformed_result], format="csr")
    return transformed_result


model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)


@wexec.willump_execute(num_threads=1)
def stack_sparse_then_linear_regression(array_one, array_two, input_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = input_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    predicted_result = model.predict(combined_result)
    return predicted_result


tf_idf_vec = \
    TfidfVectorizer(analyzer='char', ngram_range=(2, 5), vocabulary=simple_vocab_dict,
                    lowercase=False)
tf_idf_vec.fit(["theaancatdog house", "bobthe builder", "dogisgooddog"])


@wexec.willump_execute()
def stack_sparse_tfidf(array_one, array_two, input_vect, tf_idf_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = tf_idf_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    return combined_result


@wexec.willump_execute(num_threads=1)
def stack_sparse_then_linear_regression_tfidf(array_one, array_two, input_vect, tf_idf_vect):
    transformed_result_one = input_vect.transform(array_one)
    transformed_result_two = tf_idf_vect.transform(array_two)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    predicted_result = model.predict(combined_result)
    return predicted_result


class StackingNodeTests(unittest.TestCase):
    def test_sparse_stacking(self):
        print("\ntest_sparse_stacking")
        string_array = ["theaancatdog house", "bobthe builder"]
        transformed_result = vectorizer.transform(string_array)
        correct_result = scipy.sparse.hstack([transformed_result, transformed_result], format="csr").toarray()
        sample_stack_sparse(string_array, vectorizer)
        sample_stack_sparse(string_array, vectorizer)
        row, col, data, l, w = sample_stack_sparse(string_array, vectorizer)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, correct_result)

    def test_sparse_stacking_linear_model(self):
        print("\ntest_sparse_stacking_linear_model")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the"]
        result_one = vectorizer.transform(array_one)
        result_two = vectorizer.transform(array_two)
        correct_result = model.predict(scipy.sparse.hstack([result_one, result_two], format="csr"))
        stack_sparse_then_linear_regression(array_one, array_two, vectorizer)
        stack_sparse_then_linear_regression(array_one, array_two, vectorizer)
        weld_output = stack_sparse_then_linear_regression(array_one, array_two, vectorizer)
        numpy.testing.assert_equal(weld_output, correct_result)

    def test_sparse_stacking_tfidf(self):
        print("\ntest_sparse_stacking_tfidf")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog", "bbbbb"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the", "bb"]
        result_one = vectorizer.transform(array_one)
        result_two = tf_idf_vec.transform(array_two)
        correct_result = scipy.sparse.hstack([result_one, result_two], format="csr").toarray()
        stack_sparse_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        stack_sparse_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        row, col, data, l, w = stack_sparse_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        weld_matrix = scipy.sparse.csr_matrix((data, (row, col)), shape=(l, w)).toarray()
        numpy.testing.assert_almost_equal(weld_matrix, correct_result)

    def test_sparse_stacking_linreg_tfidf(self):
        print("\ntest_sparse_stacking_linreg_tfidf")
        model.coef_ = numpy.array([[0, 0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.2, 0.3, 0.4, -0.5, 0.6]], dtype=numpy.float64)
        array_one = ["dogdogdogdog house", "bobthe builder", "dog the the the the", "dog", "bbbbb"]
        array_two = ["dogdogdogdog house", "bobthe builder", "dog the the the", "dogthethe the the the the the", "bb"]
        result_one = vectorizer.transform(array_one)
        result_two = tf_idf_vec.transform(array_two)
        correct_result = model.predict(scipy.sparse.hstack([result_one, result_two], format="csr"))
        stack_sparse_then_linear_regression_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        stack_sparse_then_linear_regression_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        weld_output = stack_sparse_then_linear_regression_tfidf(array_one, array_two, vectorizer, tf_idf_vec)
        numpy.testing.assert_equal(weld_output, correct_result)
