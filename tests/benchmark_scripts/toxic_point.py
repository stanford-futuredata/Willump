# Original source here: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
import argparse
import pickle
import time

import pandas as pd
import scipy.sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from toxic_utils import *
from willump.evaluation.willump_executor import willump_execute

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-w", "--workers", type=int, help="Number of Workers")
args = parser.parse_args()
if args.cascades:
    cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
else:
    cascades = None
if args.workers is None:
    workers = 0
else:
    workers = args.workers


@willump_execute(disable=args.disable, num_workers=workers, eval_cascades=cascades)
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
_, valid_text, _, valid_target = train_test_split(train_text, train_target, test_size=0.2, random_state=42)
set_size = len(valid_text)

entry_list = []
for i in range(len(valid_text)):
    entry_list.append([valid_text[i]])

mini_text = valid_text[:2]
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)
vectorizer_transform(mini_text, word_vectorizer, char_vectorizer)
y_preds = []
times = []
for entry in tqdm(entry_list):
    t0 = time.time()
    preds = vectorizer_transform(entry, word_vectorizer, char_vectorizer)
    time_elapsed = time.time() - t0
    times.append(time_elapsed)
    y_preds.append(preds)
times = np.array(times) * 1000000

p50 = np.percentile(times, 50)
p99 = np.percentile(times, 99)
print("p50 Latency: %f us p99 Latency: %f us" %
      (p50, p99))
pickle.dump(times, open("latencies.pk", "wb"))

y_preds = np.hstack(y_preds)
print("Validation ROC-AUC Score: %f" % willump_score_function(valid_target, y_preds))
