from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

NUM_PARTITIONS = 5


def model_prediction(X, model):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64).reshape(-1, 1)
    else:
        return model.predict_proba(X)[:, 1].reshape(-1, 1)


def willump_train_function(X, y):
    lrr = LogisticRegression(solver='liblinear', penalty='l1', C=0.1)
    lrr.fit(X, y)
    return lrr


def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        return model.predict(X)


def willump_predict_proba_function(model, X):
    return model.predict_proba(X)[:, 1]


def willump_score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
