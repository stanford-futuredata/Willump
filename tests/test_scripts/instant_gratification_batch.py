import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from instant_gratification_utils import *
from willump.evaluation.willump_executor import willump_execute

base_path = "tests/test_resources/instant_gratification/"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    assert(not args.disable)
    cascades = pickle.load(open(base_path + "cascades.pk", "rb"))
    cascade_threshold = args.cascades


@willump_execute(disable=args.disable, eval_cascades=cascades, cascade_threshold=cascade_threshold)
def predict_stacked_model(X, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc):
    pred_svnu = model_prediction(X, clf_svnu)
    pred_knn = model_prediction(X, clf_knn)
    pred_lr = model_prediction(X, clf_lr)
    pred_mlp = model_prediction(X, clf_mlp)
    pred_svc = model_prediction(X, clf_svc)
    combined_pred = np.hstack([pred_svnu, pred_knn, pred_lr, pred_mlp, pred_svc])
    preds = willump_predict_function(model, combined_pred)
    return preds


if __name__ == "__main__":
    data = pd.read_csv(base_path + "train.csv")
    data = data[data['wheezy-copper-turtle-magic'] < NUM_PARTITIONS].reset_index(drop=True)
    _, valid_data = train_test_split(data, test_size=0.5, random_state=42)
    print("Validation Data Length: %d" % len(valid_data))
    valid_y = valid_data.pop('target')
    del data

    cols = [c for c in valid_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

    pca_scaler, standard_scaler = pickle.load(open(base_path + "scaler.pk", "rb"))

    valid_data = standard_scaler.transform(pca_scaler.transform(valid_data[cols]))

    clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc = pickle.load(open(base_path + "clf.pk", "rb"))
    model = pickle.load(open(base_path + "model.pk", "rb"))

    mini_data = valid_data[:3]
    predict_stacked_model(mini_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    predict_stacked_model(mini_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)

    t0 = time.time()
    preds_y = predict_stacked_model(valid_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    time_elapsed = time.time() - t0

    score = willump_score_function(valid_y, preds_y)

    print("Time elapsed:  %f sec  AUC Score: %f  Throughput: %f rows/sec" % (time_elapsed, score, len(valid_data) / time_elapsed))
