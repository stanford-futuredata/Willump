from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

NUM_PARTITIONS = 5


def model_prediction(model, X):
    return model.predict_proba(X)[:, 1].reshape(-1, 1)


def willump_train_function(X, y):
    lrr = LogisticRegression(solver='liblinear', penalty='l1', C=0.1)
    lrr.fit(X, y)
    return lrr


def willump_predict_function(model, X):
    return model.predict(X)


def willump_predict_proba_function(model, X):
    return model.predict_proba(X)[:, 1]


def willump_score_function(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
