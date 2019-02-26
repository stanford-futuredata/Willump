import argparse
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split

from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k_cascade", type=int, help="Top-K to return")
args = parser.parse_args()
if args.top_k_cascade is None:
    cascades = None
    top_K = None
else:
    cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
    top_K = args.top_k_cascade


@willump_execute(eval_cascades=cascades, top_k=top_K)
def vectorizer_transform(input_text, word_vect, char_vect):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    preds = classifier.predict_proba(combined_features)[:, 1]
    return preds


train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values
classifier = pickle.load(open(base_path + "model.pk", "rb"))
word_vectorizer, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))

class_name = "toxic"
train_target = train[class_name]
_, valid_text, _, valid_target = train_test_split(train_text, train_target, test_size=0.33, random_state=42)
del train_text, train_target
set_size = len(valid_text)

mini_text = valid_text[:2]
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)

t0 = time.time()
y_preds = vectorizer_transform(valid_text, word_vectorizer, char_vectorizer)
time_elapsed = time.time() - t0
print("Classification Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))

top_k_idx = np.argsort(y_preds)[-1 * top_K:]
top_k_values = [y_preds[i] for i in top_k_idx]

for idx, value in zip(top_k_idx, top_k_values):
    print(idx, value)
