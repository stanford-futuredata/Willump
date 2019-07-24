# Original source: https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
# Data files can be found on Kaggle:  https://www.kaggle.com/c/mercari-price-suggestion-challenge
# They must be stripped of non-ascii characters as Willump does not yet support arbirary Unicode.


import argparse
import pickle
import time
from contextlib import contextmanager
from operator import itemgetter
from typing import List, Dict, Union

import pandas as pd
import scipy.sparse
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import mercari_price_suggestion_utils
from mercari_price_suggestion_utils import *
from willump.evaluation.willump_executor import willump_execute

base_folder = "tests/test_resources/mercari_price_suggestion/"

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k_cascade", type=int, help="Top-K to return")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.top_k_cascade is None:
    cascades = None
    top_K = None
else:
    cascades = pickle.load(open(base_folder + "mercari_cascades.pk", "rb"))
    cascades["big_model"] = load_model(base_folder + "mercari_big_model.h5")
    cascades["small_model"] = load_model(base_folder + "mercari_small_model.h5")
    top_K = args.top_k_cascade


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


# model = load_model(base_folder + "mercari_big_model.h5")
model = load_model(base_folder + "mercari_model.h5")


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K)
def predict_from_input(model_input, name_vectorizer, text_vectorizer, dict_vectorizer):
    model_input = preprocess(model_input)
    name_input = model_input["name"].values
    name_vec = name_vectorizer.transform(name_input)
    text_input = model_input["text"].values
    text_vec = text_vectorizer.transform(text_input)
    valid_records = to_records(model_input[["shipping", "item_condition_id"]])
    dict_vec = dict_vectorizer.transform(valid_records)
    combined_vec = scipy.sparse.hstack([text_vec, name_vec, dict_vec], format="csr")
    preds = willump_predict_function(model, combined_vec)
    return preds


def main():
    y_scaler = StandardScaler()
    train = pd.read_table(base_folder + 'train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]
    y_scaler.fit_transform(np.log1p(train['price'].values.reshape(-1, 1)))
    mercari_price_suggestion_utils.y_scaler = y_scaler
    vectorizers = pickle.load(open(base_folder + "mercari_vect_lr.pk", "rb"))
    mini_valid = valid.iloc[0:3].copy()
    predict_from_input(mini_valid, *vectorizers)
    predict_from_input(mini_valid, *vectorizers)
    t0 = time.time()
    y_pred = np.hstack(predict_from_input(valid, *vectorizers))
    time_elapsed = time.time() - t0

    top_k_idx = np.argsort(y_pred)[-1 * top_K:]
    top_k_values = [y_pred[i] for i in top_k_idx]
    sum_values = 0

    for idx, value in zip(top_k_idx, top_k_values):
        print(idx, value)
        sum_values += value

    print("Sum of top %d values: %f" % (top_K, sum_values))

    print("Top-K Prediction Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, len(valid), len(valid) / time_elapsed))

    out_filename = "mercari_top%d.pk" % top_K
    pickle.dump(y_pred, open(out_filename, "wb"))


if __name__ == '__main__':
    main()
