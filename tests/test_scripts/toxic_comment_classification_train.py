# Original Source:  https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

import argparse
import gc
import pickle
import time

import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from toxic_comment_classification_utils import *
from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return")
args = parser.parse_args()

training_cascades = {}


@willump_execute(training_cascades=training_cascades, willump_train_function=willump_train_function,
                 willump_predict_function=willump_predict_function, willump_score_function=willump_score_function,
                 willump_predict_proba_function=willump_predict_proba_function, top_k=args.top_k)
def vectorizer_transform(input_text, word_vect, char_vect, train_target):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    clf = willump_train_function(combined_features, train_target)
    return clf


start_time = time.time()

try:
    word_vectorizer, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))
except FileNotFoundError:
    word_vectorizer = TfidfVectorizer(
        lowercase=False,
        analyzer='word',
        stop_words='english',
        ngram_range=(1, 1),
        encoding="ascii",
        decode_error="strict",
        max_features=10000)
    word_vectorizer.fit(train_text)
    char_vectorizer = TfidfVectorizer(
        lowercase=False,
        analyzer='char',
        stop_words='english',
        ngram_range=(2, 6),
        encoding="ascii",
        decode_error="strict",
        max_features=50000)
    char_vectorizer.fit(train_text)
    pickle.dump((word_vectorizer, char_vectorizer), open(base_path + "vectorizer.pk", "wb"))

class_name = "toxic"
train_target = train[class_name]
del train
gc.collect()
train_text, _, train_target, _ = train_test_split(train_text, train_target, test_size=0.2, random_state=42)

classifier = vectorizer_transform(train_text, word_vectorizer, char_vectorizer, train_target)
gc.collect()
t0 = time.time()
vectorizer_transform(train_text, word_vectorizer, char_vectorizer, train_target)
gc.collect()
print("Willump Train Time: %fs" % (time.time() - t0))

print("Total Train Time: %fs" % (time.time() - start_time))
pickle.dump(classifier, open(base_path + "model.pk", "wb"))
pickle.dump(training_cascades, open(base_path + "training_cascades.pk", "wb"))
