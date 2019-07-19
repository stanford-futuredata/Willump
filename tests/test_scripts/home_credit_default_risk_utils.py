import numpy
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def willump_train_function(X, y):
    # LightGBM parameters found by Bayesian optimization
    model = LGBMClassifier(
        n_jobs=1,
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=34,
        colsample_bytree=0.9497036,
        subsample=0.8715623,
        max_depth=8,
        reg_alpha=0.041545473,
        reg_lambda=0.0735294,
        min_split_gain=0.0222415,
        min_child_weight=39.3259775,
        silent=-1,
        verbose=-1,
        random_state=42)
    model = model.fit(X, y, eval_metric='auc', verbose=200)
    return model


def willump_predict_function(model, X):
    return model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1]


def willump_predict_proba_function(model, X):
    return model.predict_proba(X, num_iteration=model.best_iteration_)[:, 1]


def rmse_score(y, pred):
    return numpy.sqrt(mean_squared_error(y, pred))


def willump_score_function(true_y, pred_y):
    return roc_auc_score(true_y, pred_y)
