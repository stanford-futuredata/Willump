import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse
from sklearn.model_selection import train_test_split
from willump.evaluation.willump_executor import willump_execute
import argparse

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
base_path = "tests/test_resources/toxic_comment_classification/"
train = pd.read_csv(base_path + 'train.csv').fillna(' ')
train_text = train['comment_text'].values

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", help="Use cascades?", action="store_true")
args = parser.parse_args()

if args.cascades:
    training_cascades = {}
else:
    training_cascades = None


@willump_execute(training_cascades=training_cascades)
def vectorizer_transform(input_text, word_vect, char_vect, train_target):
    word_features = word_vect.transform(input_text)
    char_features = char_vect.transform(input_text)
    combined_features = scipy.sparse.hstack([word_features, char_features], format="csr")
    clf = LogisticRegression(C=0.1, solver='sag')
    clf = clf.fit(combined_features, train_target)
    return clf


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
train_text, _, train_target, _ = train_test_split(train_text, train_target, test_size=0.33, random_state=42)

vectorizer_transform(train_text, word_vectorizer, char_vectorizer, train_target)
print("First (Python) Train Done")
classifier = vectorizer_transform(train_text, word_vectorizer, char_vectorizer, train_target)
print("Second (Willump) Train Done")
pickle.dump(classifier, open(base_path + "model.pk", "wb"))
pickle.dump(training_cascades, open(base_path + "training_cascades.pk", "wb"))
