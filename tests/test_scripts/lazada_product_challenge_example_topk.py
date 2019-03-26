import argparse
import pickle
import random
import time

import numpy
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from willump.evaluation.willump_executor import willump_execute


def rmse_score(y, pred):
    return numpy.sqrt(mean_squared_error(y, pred))


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k_cascade", type=int, help="Top-K to return")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-m", "--sample", type=float, help="Sample one in S elements")
args = parser.parse_args()
if args.top_k_cascade is None:
    cascades = None
    top_K = None
else:
    cascades = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "rb"))
    top_K = args.top_k_cascade


@willump_execute(disable=args.disable, num_workers=0, eval_cascades=cascades, top_k=top_K)
def vectorizer_transform(title_vect, input_df, color_vect, brand_vect):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    brand_result = brand_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    predictions = model.predict_proba(combined_result)[:, 1]
    return predictions


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

model = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "rb"))
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

_, df, _, y = train_test_split(df, y, test_size=0.33, random_state=42)

title_vectorizer, color_vectorizer, brand_vectorizer = pickle.load(
    open("tests/test_resources/lazada_challenge_features/lazada_vectorizers.pk", "rb"))
print("Title Vocabulary has length %d" % len(title_vectorizer.vocabulary_))
print("Color Vocabulary has length %d" % len(color_vectorizer.vocabulary_))
print("Brand Vocabulary has length %d" % len(brand_vectorizer.vocabulary_))

set_size = len(df)
mini_df = df.head(2).copy()["title"]
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
t0 = time.time()
preds = vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, brand_vectorizer)
time_elapsed = time.time() - t0
print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))

if args.sample is not None:
    assert args.disable
    for i in range(len(preds)):
        if random.uniform(0, args.sample) > 1:
            preds[i] = 0

top_k_idx = np.argsort(preds)[-1 * top_K:]
top_k_values = [preds[i] for i in top_k_idx]
sum_values = 0

for idx, value in zip(top_k_idx, top_k_values):
    print(idx, value)
    sum_values += value

print("Sum of top %d values: %f" % (top_K,  sum_values))

out_filename = "lazada_top%d.pk" % top_K
pickle.dump(preds, open(out_filename, "wb"))
