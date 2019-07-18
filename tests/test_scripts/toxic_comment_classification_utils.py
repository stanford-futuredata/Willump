from sklearn.linear_model import LogisticRegression
import numpy as np


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
    return model.predict_proba(X)[:, 1]
