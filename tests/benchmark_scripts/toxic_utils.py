from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_auc_score


def willump_train_function(X, y):
    model = LogisticRegression(C=0.1, solver='sag')
    model = model.fit(X, y)
    return model


def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        return model.predict(X)


def willump_predict_proba_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    else:
        return model.predict_proba(X)[:, 1]


def willump_score_function(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)
