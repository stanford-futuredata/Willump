import argparse
import pickle
import time

import pandas as pd
import scipy.sparse
import scipy.sparse.csr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lazada_product_challenge_utils import *
from willump.evaluation.willump_executor import willump_execute

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-w", "--workers", type=int, help="Number of Workers")
args = parser.parse_args()
if args.cascades:
    cascades = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "rb"))
else:
    cascades = None

if args.workers is None:
    workers = 0
else:
    workers = args.workers


@willump_execute(disable=args.disable, num_workers=workers, eval_cascades=cascades)
def vectorizer_transform(title_vect, input_df, color_vect, brand_vect):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    brand_result = brand_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    predictions = willump_predict_function(model, combined_result)
    return predictions


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

model = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "rb"))
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

_, df, _, y = train_test_split(df, y, test_size=0.2, random_state=42)

title_vectorizer, color_vectorizer, brand_vectorizer = pickle.load(
    open("tests/test_resources/lazada_challenge_features/lazada_vectorizers.pk", "rb"))
print("Title Vocabulary has length %d" % len(title_vectorizer.vocabulary_))
print("Color Vocabulary has length %d" % len(color_vectorizer.vocabulary_))
print("Brand Vocabulary has length %d" % len(brand_vectorizer.vocabulary_))
set_size = len(df)
mini_df = df.loc[df.index[0]:df.index[0]].copy()["title"]
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer, brand_vectorizer)
entry_list = []
for i in df.index:
    entry_list.append(df.loc[i:i]["title"])
times = []
for entry in tqdm(entry_list):
    t0 = time.time()
    preds = vectorizer_transform(title_vectorizer, entry, color_vectorizer, brand_vectorizer)
    time_elapsed = time.time() - t0
    times.append(time_elapsed)

times = numpy.array(times) * 1000000

p50 = numpy.percentile(times, 50)
p99 = numpy.percentile(times, 99)
print("p50 Latency: %f us p99 Latency: %f us" %
      (p50, p99))
pickle.dump(times, open("latencies.pk", "wb"))