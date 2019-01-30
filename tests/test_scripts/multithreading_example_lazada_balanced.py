from sklearn.feature_extraction.text import CountVectorizer
import time
import pandas as pd
import willump.evaluation.willump_executor
import pickle
import numpy
import scipy.sparse.csr
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse


@willump.evaluation.willump_executor.willump_execute(num_workers=1)
def vectorizer_transform(title_vect, input_df, color_vect):
    np_input = list(input_df.values)
    transformed_result_one = title_vect.transform(np_input)
    transformed_result_two = title_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result_one, transformed_result_two], format="csr")
    predictions = model.predict(combined_result)
    return predictions


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

model = pickle.load(open("tests/test_resources/lazada_challenge_features/multi_lazada_model.pk", "rb"))
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

title_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                                   lowercase=False, stop_words=None, binary=False, decode_error='replace')
title_vectorizer.fit(df["title"].tolist())
print("Vocabulary has length %d" % len(title_vectorizer.vocabulary_))

colors = [x.strip() for x in open("tests/test_resources/lazada_challenge_features/colors.txt", encoding="windows-1252").readlines()]
c = list(filter(lambda x: len(x.split()) > 1, colors))
c = list(map(lambda x: x.replace(" ", ""), c))
colors.extend(c)

color_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4, 4), decode_error='replace', lowercase=False)
color_vectorizer.fit(colors)
print("Color Vocabulary has length %d" % len(color_vectorizer.vocabulary_))

set_size = len(df)
mini_df = df.loc[0:0].copy()["title"]
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer)
vectorizer_transform(title_vectorizer, mini_df, color_vectorizer)
entry_list = []
for i in range(set_size):
    entry_list.append(df.loc[i:i]["title"])
t0 = time.time()
for entry in entry_list:
    preds = vectorizer_transform(title_vectorizer, entry, color_vectorizer)
time_elapsed = time.time() - t0
print("Title Processing Time %fs Num Rows %d Latency %f sec/row" %
      (time_elapsed, set_size, time_elapsed / set_size))
