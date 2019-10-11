import unittest
import numpy as np
from willump.evaluation.willump_executor import willump_execute
from sklearn.metrics import roc_auc_score
import time


def willump_train_function(X, y):
    return range(min(X.shape[1], 2))


# Predict the sum of the first "model" columns unless it is zero, then predict the value of the last column.
def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    else:
        x_sum = sum(X[:, i] for i in model)
        for i, val in enumerate(x_sum):
            if val == 0.0:
                x_sum[i] = X[i, -1]
        return (x_sum > 0.5).astype(np.int64)


# Confidence is the sum of the first "model" columns, unless that sum is 0, then confidence is 0.5.
def willump_predict_proba_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    else:
        x_sum = sum(X[:, i] for i in model)
        for i, val in enumerate(x_sum):
            if val == 0.0:
                x_sum[i] = 0.5
        return x_sum


def willump_score_function(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)


cascades = {}


def feature_one_basic(X):
    time.sleep(0.1)
    return X[:, 0].reshape(-1, 1)


def feature_two_basic(X):
    time.sleep(0.1)
    return X[:, 1].reshape(-1, 1)


def feature_three_basic(X):
    time.sleep(0.25)
    return X[:, 2].reshape(-1, 1)


@willump_execute(training_cascades=cascades, willump_train_function=willump_train_function,
                 willump_predict_function=willump_predict_function,
                 willump_predict_proba_function=willump_predict_proba_function,
                 willump_score_function=willump_score_function)
def basic_cascades_train(X, y):
    f1 = feature_one_basic(X)
    f2 = feature_two_basic(X)
    f3 = feature_three_basic(X)
    fc = np.hstack([f1, f2, f3])
    model = willump_train_function(fc, y)
    return model


@willump_execute(eval_cascades=cascades)
def basic_cascades_eval(model, X):
    f1 = feature_one_basic(X)
    f2 = feature_two_basic(X)
    f3 = feature_three_basic(X)
    fc = np.hstack([f1, f2, f3])
    preds = willump_predict_function(model, fc)
    return preds


class WillumpCascadesTest(unittest.TestCase):
    def test_cascade_one_train(self):
        print("\ntest_cascade_train")
        X = np.random.rand(1000, 3)
        X[:950, :2] /= 2
        X[950:, :2] = 0
        y = willump_predict_function(range(2), X)
        basic_cascades_train(X, y)
        basic_cascades_train(X, y)
        assert (cascades["cascade_threshold"] == 0.6)
        assert (cascades["cost_cutoff"] == 0.5)

    def test_cascade_two_eval(self):
        print("\ntest_cascade_eval")
        X = np.random.rand(1000, 3)
        X[:950, :2] /= 2
        X[950:, :2] = 0
        y = willump_predict_function(range(2), X)
        preds_py = basic_cascades_eval(cascades["big_model"], X)
        np.testing.assert_equal(preds_py, y)
        preds_willump = basic_cascades_eval(cascades["big_model"], X)
        np.testing.assert_equal(preds_willump, y)
