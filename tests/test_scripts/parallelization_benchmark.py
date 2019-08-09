import argparse
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-w", "--workers", type=int, help="Number of Workers")
args = parser.parse_args()
if args.workers is None:
    workers = 0
else:
    workers = args.workers


def willump_predict_function(model, X):
    return model.predict(X)


@willump_execute(disable=args.disable, num_workers=workers)
def vectorizer_transform(input_text, char_vect):
    char_features_one = char_vect.transform(input_text)
    char_features_two = char_vect.transform(input_text)
    char_features_three = char_vect.transform(input_text)
    char_features_four = char_vect.transform(input_text)
    char_features_five = char_vect.transform(input_text)
    char_features_six = char_vect.transform(input_text)
    char_features_seven = char_vect.transform(input_text)
    char_features_eight = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([char_features_one, char_features_two,
                                             char_features_three, char_features_four,
                                             char_features_five, char_features_six,
                                             char_features_seven, char_features_eight], format="csr")
    preds = willump_predict_function(classifier, combined_features)
    return preds


train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values
classifier = pickle.load(open(base_path + "model.pk", "rb"))
_, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))
classifier.coef_ = np.random.rand(len(char_vectorizer.vocabulary_) * 8).reshape(1, -1)

_, valid_text = train_test_split(train_text, test_size=0.1, random_state=42)
set_size = len(valid_text)

entry_list = []
for i in range(len(valid_text)):
    entry_list.append([valid_text[i]])

mini_text = valid_text[:2]
vectorizer_transform(mini_text, char_vectorizer)
vectorizer_transform(mini_text, char_vectorizer)
times = []
for entry in tqdm(entry_list):
    t0 = time.time()
    vectorizer_transform(entry, char_vectorizer)
    time_elapsed = time.time() - t0
    times.append(time_elapsed)
p50 = np.percentile(times, 50)
p99 = np.percentile(times, 99)

print("p50 Latency: %f p99 Latency: %f" %
      (p50, p99))
