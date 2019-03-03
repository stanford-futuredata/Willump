# Original source: https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
# Data files can be found on Kaggle:  https://www.kaggle.com/c/mercari-price-suggestion-challenge
# They must be stripped of non-ascii characters as Willump does not yet support arbirary Unicode.


import pickle
import time
from contextlib import contextmanager
from operator import itemgetter
from typing import List, Dict, Union

import keras as ks
import numpy as np
import pandas as pd
import scipy.sparse
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

base_folder = "tests/test_resources/mercari_price_suggestion/"


def fit_neural_net(X_train, y_train) -> np.ndarray:
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1)(out)
        model = ks.Model(model_in, out)
        model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(1):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2 ** (11 + i), epochs=1, verbose=0)
        model.save(base_folder + "mercari_model.h5")


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


def process_input(model_input, name_vectorizer, text_vectorizer, dict_vectorizer):
    model_input = preprocess(model_input)
    name_input = model_input["name"].values
    name_vec = name_vectorizer.transform(name_input)
    text_input = model_input["text"].values
    text_vec = text_vectorizer.transform(text_input)
    valid_records = to_records(model_input[["shipping", "item_condition_id"]])
    dict_vec = dict_vectorizer.transform(valid_records)
    combined_vec = scipy.sparse.hstack([name_vec, text_vec, dict_vec], format="csr")
    return combined_vec


def main():
    y_scaler = StandardScaler()
    train = pd.read_table(base_folder + 'train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]
    y_train = y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    try:
        vectorizers = pickle.load(open(base_folder + "mercari_vect_lr.pk", "rb"))
    except FileNotFoundError:
        with timer('create vectorizers'):
            vectorizers = create_vectorizers(train)
            pickle.dump(vectorizers, open(base_folder + "mercari_vect_lr.pk", "wb"))
    with timer('Process Train Input'):
        X_train = process_input(train, *vectorizers).astype(np.float32)
    with timer("Train model"):
        fit_neural_net(X_train, y_train)


if __name__ == '__main__':
    main()
