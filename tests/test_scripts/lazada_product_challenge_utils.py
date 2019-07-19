from sklearn.linear_model import LogisticRegression
import numpy
from sklearn.metrics import mean_squared_error


def willump_train_function(X, y):
    model = LogisticRegression()
    model = model.fit(X, y)
    return model


def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return numpy.zeros(0, dtype=numpy.int64)
    else:
        return model.predict(X)


def willump_predict_proba_function(model, X):
    return model.predict_proba(X)[:, 1]


def rmse_score(y, pred):
    return numpy.sqrt(mean_squared_error(y, pred))


def willump_score_function(true_y, pred_y):
    return 1 - rmse_score(true_y, pred_y)
