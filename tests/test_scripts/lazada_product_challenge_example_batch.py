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


def rmse_score(y, pred):
    return numpy.sqrt(mean_squared_error(y, pred))


@willump.evaluation.willump_executor.willump_execute()
def vectorizer_transform(input_vect, input_df, tf_idf_vect):
    np_input = list(input_df.values)
    transformed_result = input_vect.transform(np_input)
    color_result = tf_idf_vect.transform(np_input)
    combined_result = scipy.sparse.hstack([transformed_result, color_result], format="csr")
    predictions = model.predict(combined_result)
    return predictions


df = pd.read_csv("tests/test_resources/lazada_challenge_features/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

model = pickle.load(open("tests/test_resources/lazada_challenge_features/lazada_model.pk", "rb"))
y = numpy.loadtxt("tests/test_resources/lazada_challenge_features/conciseness_train.labels", dtype=int)

vect = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                       lowercase=False, stop_words=None, binary=False, decode_error='replace')
vect.fit(df["title"].tolist())
print("Vocabulary has length %d" % len(vect.vocabulary_))

colors = [x.strip() for x in open("tests/test_resources/lazada_challenge_features/colors.txt", encoding="windows-1252").readlines()]
c = list(filter(lambda x: len(x.split()) > 1, colors))
c = list(map(lambda x: x.replace(" ", ""), c))
colors.extend(c)

tf_idf_vec = TfidfVectorizer(analyzer='char', ngram_range=(4, 4), decode_error='replace', lowercase=False)
tf_idf_vec.fit(colors)
print("Color Vocabulary has length %d" % len(tf_idf_vec.vocabulary_))

set_size = len(df)
mini_df = df.head(2).copy()["title"]
vectorizer_transform(vect, mini_df, tf_idf_vec)
vectorizer_transform(vect, mini_df, tf_idf_vec)
vectorizer_transform(vect, mini_df, tf_idf_vec)
t0 = time.time()
preds = vectorizer_transform(vect, df["title"], tf_idf_vec)
time_elapsed = time.time() - t0
print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))

print("RMSE Score: %f" % rmse_score(preds, y))
