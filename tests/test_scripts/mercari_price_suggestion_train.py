# Original source: https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
# Data files can be found on Kaggle:  https://www.kaggle.com/c/mercari-price-suggestion-challenge
# They must be stripped of non-ascii characters as Willump does not yet support arbirary Unicode.


import pickle
import time
from contextlib import contextmanager
from operator import itemgetter
from typing import List, Dict, Union

import argparse
import keras as ks
import pandas as pd
import scipy.sparse
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import mercari_price_suggestion_utils
from mercari_price_suggestion_utils import *
from willump.evaluation.willump_executor import willump_execute

base_folder = "tests/test_resources/mercari_price_suggestion/"

config = tf.ConfigProto(
    intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return")
parser.add_argument("-b", "--debug", help="Debug Mode", action="store_true")
args = parser.parse_args()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = (df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna(''))
    return df[['name', 'text', 'shipping', 'item_condition_id']]


def on_field(f: Union[str, List[str]], *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')


def create_vectorizers(train):
    train = preprocess(train)
    name_vectorizer = Tfidf(max_features=100000, lowercase=False, encoding="ascii", decode_error="strict",
                            analyzer="word")
    name_vectorizer.fit(train["name"])
    text_vectorizer = Tfidf(max_features=100000, ngram_range=(1, 1), lowercase=False, encoding="ascii",
                            decode_error="strict", analyzer="word")  # TODO:  ngram_range = (1, 2) after segfault fixed.
    text_vectorizer.fit(train["text"])
    valid_records = to_records(train[["shipping", "item_condition_id"]])
    dict_vectorizer = DictVectorizer()
    dict_vectorizer.fit(valid_records)
    return name_vectorizer, text_vectorizer, dict_vectorizer


training_cascades = {}


@willump_execute(training_cascades=training_cascades, willump_train_function=willump_train_function,
                 willump_predict_function=willump_predict_function,
                 willump_predict_proba_function=willump_predict_proba_function,
                 willump_score_function=willump_score_function,
                 top_k=args.top_k)
def process_input_and_train(model_input, name_vectorizer, text_vectorizer, dict_vectorizer, y_train):
    model_input = preprocess(model_input)
    name_input = model_input["name"].values
    name_vec = name_vectorizer.transform(name_input)
    text_input = model_input["text"].values
    text_vec = text_vectorizer.transform(text_input)
    valid_records = to_records(model_input[["shipping", "item_condition_id"]])
    dict_vec = dict_vectorizer.transform(valid_records)
    combined_vec = scipy.sparse.hstack([name_vec, dict_vec, text_vec], format="csr")
    model = willump_train_function(combined_vec, y_train)
    return model


def main():
    global sess
    y_scaler = StandardScaler()
    train = pd.read_table(base_folder + 'train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    train_ids, _ = next(cv.split(train))
    if args.debug:
        train_ids = train_ids[:10000]
    train = train.iloc[train_ids]
    print("Training set rows %d" % len(train))
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    mercari_price_suggestion_utils.y_scaler = y_scaler
    try:
        vectorizers = pickle.load(open(base_folder + "mercari_vect_lr.pk", "rb"))
    except FileNotFoundError:
        with timer('create vectorizers'):
            vectorizers = create_vectorizers(train)
            pickle.dump(vectorizers, open(base_folder + "mercari_vect_lr.pk", "wb"))

    sess = tf.Session(config=config)
    ks.backend.set_session(sess)
    model = process_input_and_train(train.copy(), *vectorizers, y_train)
    model.save(base_folder + "mercari_model.h5")
    sess.close()
    sess = tf.Session(config=config)
    ks.backend.set_session(sess)
    with timer('Second (Willump) Training'):
        process_input_and_train(train.copy(), *vectorizers, y_train)
    training_cascades["big_model"].save(base_folder + "mercari_big_model.h5")
    training_cascades["small_model"].save(base_folder + "mercari_small_model.h5")
    pickle.dump(training_cascades, open(base_folder + "mercari_cascades.pk", "wb"))
    sess.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("Total Train Time: %fs" % (time.time() - start_time))
