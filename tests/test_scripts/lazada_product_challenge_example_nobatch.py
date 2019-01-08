from sklearn.feature_extraction.text import CountVectorizer
import time
import pandas as pd
import willump.evaluation.willump_executor
import scipy.sparse.csr


@willump.evaluation.willump_executor.willump_execute
def vectorizer_transform(input_vect, input_df):
    np_input = list(input_df.values)
    transformed_result = input_vect.transform(np_input)
    return transformed_result


df = pd.read_csv("tests/test_resources/lazada_data_train.csv", header=None,
                 names=['country', 'sku_id', 'title', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3',
                        'short_description', 'price', 'product_type'])

vect = CountVectorizer(analyzer='char', ngram_range=(2, 6), min_df=0.005, max_df=1.0,
                       lowercase=False, stop_words=None, binary=False, decode_error='replace')
vect.fit(df["title"].tolist())
print("Vocabulary has length %d" % len(vect.vocabulary_))

set_size = len(df)
mini_df = df.loc[0:0].copy()["title"]
vectorizer_transform(vect, mini_df)
vectorizer_transform(vect, mini_df)
vectorizer_transform(vect, mini_df)
entry_list = []
for i in range(set_size):
    entry_list.append(df.loc[i:i]["title"])
t0 = time.time()
for entry in entry_list:
    X_title = vectorizer_transform(vect, entry)
time_elapsed = time.time() - t0
print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))
