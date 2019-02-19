import unittest
import importlib
import numpy
import scipy.sparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import willump.evaluation.willump_executor as wexec
import willump.evaluation.willump_weld_generator

from willump.graph.willump_graph import WillumpGraph
from willump.graph.willump_input_node import WillumpInputNode
from willump.graph.willump_output_node import WillumpOutputNode
from willump.graph.array_tfidf_node import ArrayTfIdfNode
from weld.types import *
import sklearn.linear_model

with open("tests/test_resources/simple_vocabulary.txt") as simple_vocab:
    simple_vocab_dict = {word: index for index, word in
                         enumerate(simple_vocab.read().splitlines())}
tf_idf_vec_char = \
    TfidfVectorizer(analyzer='char', ngram_range=(2, 6),
                    lowercase=False)

tf_idf_vec_word = \
    TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                    lowercase=False)


@wexec.willump_execute()
def sample_tfidf_vectorizer(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


@wexec.willump_execute()
def sample_tfidf_vectorizer_word(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


model = sklearn.linear_model.LogisticRegression(solver='lbfgs')
model.intercept_ = numpy.array([0.2], dtype=numpy.float64)
model.classes_ = numpy.array([0, 1], dtype=numpy.int64)


@wexec.willump_execute()
def tf_idf_vectorizer_then_regression(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    prediction = model.predict(transformed_result)
    return prediction


@wexec.willump_execute()
def tf_idf_vectorizer_then_regression_word(array_one, input_vect):
    df = pd.DataFrame()
    df["strings"] = array_one
    np_input = list(df["strings"].values)
    transformed_result = input_vect.transform(np_input)
    prediction = model.predict(transformed_result)
    return prediction


class TfidfNodeTests(unittest.TestCase):
    def setUp(self):
        self.input_str = [
            " catdogcat", "the dog house ", "elephantcat", "Bbbbbbb", "an ox cat house", "ananan",
            "      an ox cat house   ",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"  # TODO:  Talk to Shoumik about this.
            "Summer new British printing round neck short sleeve male T-shirt(black)-intl",
            "UniSilver TIME Lunarchia Pair Women's Silver / Pink Mother-of-Pearl Analog Stainless Steel Watch KW1367-2909",
            "Foldable Selfie Stick Monopod (Green) With 3 in 1 Macro/Fish-eye/Wide Clip Lens for Mobile Phone and Tablets (Green)"
            "on the server Farstriders."
            "Where do you come from?"
            "\"\nTell that stuff to AMIB, will ya!?  \"",
        ]
        tf_idf_vec_char.fit(self.input_str)
        tf_idf_vec_word.fit(self.input_str)
        self.idf_vec = tf_idf_vec_char.idf_
        self.correct_output = tf_idf_vec_char.transform(self.input_str).toarray()
        self.correct_output_word = tf_idf_vec_word.transform(self.input_str).toarray()

    def test_infer_tfidf_vectorizer(self):
        print("\ntest_infer_tfidf_vectorizer")
        sample_tfidf_vectorizer(self.input_str, tf_idf_vec_char)
        sample_tfidf_vectorizer(self.input_str, tf_idf_vec_char)
        weld_csr_matrix = sample_tfidf_vectorizer(self.input_str, tf_idf_vec_char)
        weld_matrix = weld_csr_matrix.toarray()
        numpy.testing.assert_almost_equal(weld_matrix, self.correct_output)

    def test_infer_tfidf_vectorizer_word(self):
        print("\ntest_infer_tfidf_vectorizer_word")
        sample_tfidf_vectorizer_word(self.input_str, tf_idf_vec_word)
        sample_tfidf_vectorizer_word(self.input_str, tf_idf_vec_word)
        weld_csr_matrix = sample_tfidf_vectorizer_word(self.input_str, tf_idf_vec_word)
        weld_matrix = weld_csr_matrix.toarray()
        numpy.testing.assert_almost_equal(weld_matrix, self.correct_output_word)

    def test_linear_model_tfidf_vectorizer(self):
        print("\ntest_linear_model_tfidf_vectorizer")
        model.coef_ = numpy.random.random(len(tf_idf_vec_char.vocabulary_)).reshape(1, -1)
        tf_idf_vectorizer_then_regression(self.input_str, tf_idf_vec_char)
        tf_idf_vectorizer_then_regression(self.input_str, tf_idf_vec_char)
        preds = tf_idf_vectorizer_then_regression(self.input_str, tf_idf_vec_char)
        correct_preds = model.predict(self.correct_output)
        numpy.testing.assert_almost_equal(preds, correct_preds)

    def test_linear_model_tfidf_vectorizer_word(self):
        print("\ntest_linear_model_tfidf_vectorizer_word")
        model.coef_ = numpy.random.random(len(tf_idf_vec_word.vocabulary_)).reshape(1, -1)
        tf_idf_vectorizer_then_regression_word(self.input_str, tf_idf_vec_word)
        tf_idf_vectorizer_then_regression_word(self.input_str, tf_idf_vec_word)
        preds = tf_idf_vectorizer_then_regression_word(self.input_str, tf_idf_vec_word)
        correct_preds = model.predict(self.correct_output_word)
        numpy.testing.assert_almost_equal(preds, correct_preds)
