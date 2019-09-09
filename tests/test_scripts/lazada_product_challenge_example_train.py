# Original Source: http://ls3.rnet.ryerson.ca/wp-content/uploads/2017/10/CIKM_AnalytiCup_2017_Solution.zip

import argparse
import pickle
import time

import numpy
import pandas as pd
import scipy.sparse
import scipy.sparse.csr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import willump.evaluation.willump_executor
from lazada_product_challenge_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return")
args = parser.parse_args()

training_cascades = {}


@willump.evaluation.willump_executor.willump_execute(num_workers=0, training_cascades=training_cascades,
                                                     willump_train_function=willump_train_function,
                                                     willump_predict_function=willump_predict_function,
                                                     willump_predict_proba_function=willump_predict_proba_function,
                                                     willump_score_function=willump_score_function,
                                                     top_k=args.top_k)
def vectorizer_transform(title_vect, input_df, color_vect, brand_vect, y_df):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    brand_result = brand_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result, brand_result], format="csr")
    model = willump_train_function(combined_result, y_df)
    return model

start_time = time.time()

df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

df, _, y, _ = train_test_split(df, y, test_size=0.2, random_state=42)

title_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                                   lowercase=False, stop_words=None, binary=False, decode_error='replace')
title_vectorizer.fit(df["title"].tolist())
print("Vocabulary has length %d" % len(title_vectorizer.vocabulary_))

colors = [x.strip() for x in
          open("tests/test_resources/lazada_challenge_features/colors.txt", encoding="windows-1252").readlines()]
c = list(filter(lambda x: len(x.split()) > 1, colors))
c = list(map(lambda x: x.replace(" ", ""), c))
colors.extend(c)

color_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace', lowercase=False)
color_vectorizer.fit(colors)
print("Color Vocabulary has length %d" % len(color_vectorizer.vocabulary_))

brands = [x.strip() for x in open("tests/test_resources/lazada_challenge_features/brands_from_lazada_portal.txt",
                                  encoding="windows-1252").readlines()]
brand_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), decode_error='replace', lowercase=False)
brand_vectorizer.fit(brands)
print("Brand Vocabulary has length %d" % len(brand_vectorizer.vocabulary_))

set_size = len(df)
trained_model = vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, brand_vectorizer, y)

t0 = time.time()
vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, brand_vectorizer, y)
time_elapsed = time.time() - t0
print("Willump Training Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))

pickle.dump((title_vectorizer, color_vectorizer, brand_vectorizer),
            open("tests/test_resources/lazada_challenge_features/lazada_vectorizers.pk", "wb"))
pickle.dump(trained_model, open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "wb"))
pickle.dump(training_cascades, open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "wb"))

print("Total Train Time: %fs" % (time.time() - start_time))
