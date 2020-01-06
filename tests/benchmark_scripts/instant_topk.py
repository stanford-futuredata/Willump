# Data here:  https://www.kaggle.com/c/instant-gratification/data
# Original source here: https://www.kaggle.com/prashantkikani/ig-pca-nusvc-knn-lr-stack
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from instant_utils import *
from willump.evaluation.willump_executor import willump_execute

base_path = "tests/test_resources/instant_gratification/"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascades?")
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()

if args.cascades:
    cascades = pickle.load(open(base_path + "cascades.pk", "rb"))
else:
    cascades = None
top_K = args.top_k


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K)
def predict_stacked_model(X, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc):
    pred_svnu = model_prediction(X, clf_svnu)
    # pred_knn = model_prediction(X, clf_knn)
    pred_lr = model_prediction(X, clf_lr)
    pred_mlp = model_prediction(X, clf_mlp)
    pred_svc = model_prediction(X, clf_svc)
    combined_pred = np.hstack([pred_svnu, pred_lr, pred_mlp, pred_svc])
    preds = willump_predict_proba_function(model, combined_pred)
    return preds


if __name__ == "__main__":
    data = pd.read_csv(base_path + "train.csv")
    _, valid_data = train_test_split(data, test_size=0.1, random_state=42)
    valid_data = valid_data[valid_data['wheezy-copper-turtle-magic'] < NUM_PARTITIONS].reset_index(drop=True)
    print("Validation Data Length: %d" % len(valid_data))
    valid_y = valid_data.pop('target')
    partition_column = valid_data.pop('wheezy-copper-turtle-magic').values.reshape(-1, 1)
    del data

    cols = [c for c in valid_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

    pca_scaler, standard_scaler = pickle.load(open(base_path + "scaler.pk", "rb"))

    valid_data = standard_scaler.transform(pca_scaler.transform(valid_data[cols]))
    valid_data = np.append(valid_data, partition_column, 1)

    clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc = pickle.load(open(base_path + "clf.pk", "rb"))

    model = pickle.load(open(base_path + "model.pk", "rb"))

    mini_data = valid_data[:3]
    orig_preds = predict_stacked_model(valid_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    predict_stacked_model(mini_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)

    t0 = time.time()
    preds = predict_stacked_model(valid_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    time_elapsed = time.time() - t0

    orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
    actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
    precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

    orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
    actual_model_sum = sum(preds[actual_model_top_k_idx])
    set_size = len(valid_data)
    print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
          (time_elapsed, set_size, set_size / time_elapsed))
    print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))
