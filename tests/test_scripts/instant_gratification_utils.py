from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

NUM_PARTITIONS = 512
PARTITION_INDEX = -1


def model_prediction(X, model):
    preds = np.zeros(len(X), dtype=np.float64)
    if X.shape[0] == 0:
        return preds.reshape(-1, 1)
    else:
        for i in range(NUM_PARTITIONS):
            partition_indices = np.nonzero(X[:, PARTITION_INDEX] == i)[0]
            if len(partition_indices) > 0:
                preds[partition_indices] = model[i].predict_proba(X[partition_indices, :-1])[:, 1]
        return preds.reshape(-1, 1)


def willump_train_function(X, y):
    lrr = LogisticRegression(solver='liblinear')
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
