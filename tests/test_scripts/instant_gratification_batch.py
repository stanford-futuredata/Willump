import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from instant_gratification_utils import *

base_path = "tests/test_resources/instant_gratification/"


def predict_stacked_model(X, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc):
    pred_svnu = model_prediction(clf_svnu, X)
    pred_knn = model_prediction(clf_knn, X)
    pred_lr = model_prediction(clf_lr, X)
    pred_mlp = model_prediction(clf_mlp, X)
    pred_svc = model_prediction(clf_svc, X)
    combined_pred = np.concatenate((pred_svnu, pred_knn, pred_lr, pred_mlp, pred_svc), axis=1)
    preds = willump_predict_proba_function(model, combined_pred)
    return preds


if __name__ == "__main__":
    data = pd.read_csv(base_path + "train.csv")
    data = data[data['wheezy-copper-turtle-magic'] < NUM_PARTITIONS].reset_index(drop=True)
    _, valid_data = train_test_split(data, test_size=0.5, random_state=42)
    print("Validation Data Length: %d" % len(valid_data))
    valid_y = valid_data.pop('target')
    del data

    cols = [c for c in valid_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

    valid_data = StandardScaler().fit_transform(
        PCA(svd_solver='full', n_components='mle').fit_transform(valid_data[cols]))

    clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc = pickle.load(open(base_path + "clf.pk", "rb"))
    model = pickle.load(open(base_path + "model.pk", "rb"))

    t0 = time.time()
    preds_y = predict_stacked_model(valid_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    time_elapsed = time.time() - t0

    score = willump_score_function(valid_y, preds_y)

    print("Time elapsed:  %d  AUC Score: %f  Throughput: %f" % (time_elapsed, score, time_elapsed / len(valid_data)))
