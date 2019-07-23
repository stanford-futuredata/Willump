import argparse
import pickle
import time

import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from toxic_comment_classification_utils import *
from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-w", "--workers", type=int, help="Number of Workers")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    assert(not args.disable)
    cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
    cascade_threshold = args.cascades
if args.workers is None:
    workers = 0
else:
    workers = args.workers


@willump_execute(disable=args.disable, num_workers=workers, eval_cascades=cascades, cascade_threshold=cascade_threshold)
def vectorizer_transform(input_text, word_vect, char_vect):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    preds = willump_predict_function(classifier, combined_features)
    return preds


train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values
classifier = pickle.load(open(base_path + "model.pk", "rb"))
word_vectorizer, char_vectorizer = pickle.load(open(base_path + "vectorizer.pk", "rb"))

class_name = "toxic"
train_target = train[class_name]
_, valid_text, _, valid_target = train_test_split(train_text, train_target, test_size=0.33, random_state=42)
set_size = len(valid_text)

entry_list = []
for i in range(len(valid_text)):
    entry_list.append([valid_text[i]])

mini_text = valid_text[:2]
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)
y_preds = []
t0 = time.time()
for entry in tqdm(entry_list):
    preds = vectorizer_transform(entry, word_vectorizer, char_vectorizer)
    y_preds.append(preds)
time_elapsed = time.time() - t0
print("Classification Time %fs Num Rows %d Throughput %f rows/sec Latency %f sec/row" %
      (time_elapsed, set_size, set_size / time_elapsed, time_elapsed / set_size))

y_preds = np.hstack(y_preds)
print("Validation ROC-AUC Score: %f" % willump_score_function(valid_target, y_preds))
