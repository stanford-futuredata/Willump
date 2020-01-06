# Original source here: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
import argparse
import pickle
import time

import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split

from toxic_utils import *
from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascades?")
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()

if args.cascades:
    cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
else:
    cascades = None
top_K = args.top_k


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K)
def vectorizer_transform(input_text, word_vect, char_vect):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    preds = willump_predict_proba_function(classifier, combined_features)
    return preds


train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values
classifier = pickle.load(open(base_path + "model.pk", "rb"))
word_vectorizer, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))

class_name = "toxic"
train_target = train[class_name]
_, valid_text, _, _ = train_test_split(train_text, train_target, test_size=0.2, random_state=42)
del train_text, train_target
set_size = len(valid_text)

mini_text = valid_text[:2]
orig_preds = vectorizer_transform(valid_text, word_vectorizer, char_vectorizer)
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)

t0 = time.time()
preds = vectorizer_transform(valid_text, word_vectorizer, char_vectorizer)
time_elapsed = time.time() - t0

orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
actual_model_sum = sum(preds[actual_model_top_k_idx])

print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
      (time_elapsed, set_size, set_size / time_elapsed))
print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))