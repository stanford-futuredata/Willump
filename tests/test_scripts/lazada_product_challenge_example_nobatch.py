import pickle
import time
import argparse

import numpy
import pandas as pd
import scipy.sparse
import scipy.sparse.csr
from sklearn.model_selection import train_test_split

import willump.evaluation.willump_executor

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    cascades = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "rb"))
    cascade_threshold = args.cascades


@willump.evaluation.willump_executor.willump_execute(num_workers=2, eval_cascades=cascades,
                                                     cascade_threshold=cascade_threshold)
def vectorizer_transform(title_vect, input_df, color_vect, brand_vect):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    brand_result = brand_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    predictions = model.predict(combined_result)
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
mini_df = df.loc[0:0].copy()["title"]
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
entry_list = []
for i in df.index:
    entry_list.append(df.loc[i:i]["title"])
t0 = time.time()
for entry in entry_list:
    preds = vectorizer_transform(title_vectorizer, entry, color_vectorizer, brand_vectorizer)
time_elapsed = time.time() - t0
print("Title Processing Time %fs Num Rows %d Latency %f sec/row" %
      (time_elapsed, set_size, time_elapsed / set_size))
