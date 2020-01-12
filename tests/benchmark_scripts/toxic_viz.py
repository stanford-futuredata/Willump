# Original source here: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
import argparse
import pickle
from multiprocessing import Process

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
    trained_cascades = pickle.load(open(base_path + "training_cascades.pk", "rb"))
else:
    trained_cascades = None
if args.workers is None:
    workers = 0
else:
    workers = args.workers









@willump_execute(eval_cascades=trained_cascades)
def vectorizer_transform(input_text, word_vect, char_vect):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    preds = willump_predict_function(classifier, combined_features)
    return preds











def vectorizer_transform_noopt(input_text, word_vect, char_vect):
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


def with_willump():
    for entry in tqdm(entry_list, position=1, desc="With Willump"):
        vectorizer_transform(entry, word_vectorizer, char_vectorizer)


def sans_willump():
    for entry in tqdm(entry_list, desc="Sans Willump"):
        vectorizer_transform_noopt(entry, word_vectorizer, char_vectorizer)


pw = Process(target=with_willump)
ps = Process(target=sans_willump)

print("Starting performance comparison!")

pw.start()
ps.start()
pw.join()
ps.join()
