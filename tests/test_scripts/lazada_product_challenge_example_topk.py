import argparse
import pickle
import random
import time

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csr
from sklearn.model_selection import train_test_split

from lazada_product_challenge_utils import *
from willump.evaluation.willump_executor import willump_execute


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascades?")
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()

if args.cascades:
    cascades = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "rb"))
else:
    cascades = None
top_K = args.top_k


@willump_execute(disable=args.disable, num_workers=0, eval_cascades=cascades, top_k=top_K)
def vectorizer_transform(title_vect, input_df, color_vect, brand_vect):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    brand_result = brand_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    predictions = willump_predict_proba_function(model, combined_result)
    return predictions


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

model = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "rb"))

_, df, = train_test_split(df, test_size=0.2, random_state=42)

title_vectorizer, color_vectorizer, brand_vectorizer = pickle.load(
    open("tests/test_resources/lazada_challenge_features/lazada_vectorizers.pk", "rb"))

set_size = len(df)
mini_df = df.head(2).copy()["title"]
orig_preds = vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
t0 = time.time()
preds = vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, brand_vectorizer)
time_elapsed = time.time() - t0

orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
actual_model_sum = sum(preds[actual_model_top_k_idx])

print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))
print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))
