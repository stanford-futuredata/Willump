from sklearn.feature_extraction.text import CountVectorizer
import time
import pandas as pd
import numpy
import pickle
import scipy.sparse
import scipy.sparse.csr
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import willump.evaluation.willump_executor

training_cascades = {}


@willump.evaluation.willump_executor.willump_execute(num_workers=0, training_cascades=training_cascades)
def vectorizer_transform(title_vect, input_df, color_vect, y_df):
    np_input = list(input_df.values)
    transformed_result = title_vect.transform(np_input)
    color_result = color_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result], format="csr")
    model = SGDClassifier(loss='log', penalty='l1', max_iter=1000, verbose=0, tol=0.0001)
    model = model.fit(combined_result, y_df)
    return model


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

title_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                                   lowercase=False, stop_words=None, binary=False, decode_error='replace')
title_vectorizer.fit(df["title"].tolist())
print("Vocabulary has length %d" % len(title_vectorizer.vocabulary_))

colors = [x.strip() for x in
          open("tests/test_resources/lazada_challenge_features/colors.txt", encoding="windows-1252").readlines()]
c = list(filter(lambda x: len(x.split()) > 1, colors))
c = list(map(lambda x: x.replace(" ", ""), c))
colors.extend(c)

color_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(4, 4), decode_error='replace', lowercase=False)
color_vectorizer.fit(colors)
print("Color Vocabulary has length %d" % len(color_vectorizer.vocabulary_))

set_size = len(df)
t0 = time.time()
vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, y)
time_elapsed = time.time() - t0
print("First (Python) Training Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))
t0 = time.time()
trained_model = vectorizer_transform(title_vectorizer, df["title"], color_vectorizer, y)
time_elapsed = time.time() - t0
print(trained_model)
print("Second (Willump Cascade) Training Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))

pickle.dump(trained_model, open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "wb"))
pickle.dump(training_cascades, open("tests/test_resources/lazada_challenge_features/lazada_training_cascades.pk", "wb"))
