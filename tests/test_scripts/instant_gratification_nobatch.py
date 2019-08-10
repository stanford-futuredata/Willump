import argparse
import pickle
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from instant_gratification_utils import *
from willump.evaluation.willump_executor import willump_execute

base_path = "tests/test_resources/instant_gratification/"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.cascades:
    cascades = pickle.load(open(base_path + "cascades.pk", "rb"))
else:
    cascades=None


@willump_execute(disable=args.disable, eval_cascades=cascades)
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
    _, valid_data = train_test_split(data, test_size=0.1, random_state=42)
    valid_data, _ = train_test_split(valid_data, test_size=0.5, random_state=42)
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

    # print("svnu: %f" % roc_auc_score(valid_y, model_prediction(valid_data, clf_svnu)))
    # print("knn: %f" % roc_auc_score(valid_y, model_prediction(valid_data, clf_knn)))
    # print("lr: %f" % roc_auc_score(valid_y, model_prediction(valid_data, clf_lr)))
    # print("mlp: %f" % roc_auc_score(valid_y, model_prediction(valid_data, clf_mlp)))
    # print("svc: %f" % roc_auc_score(valid_y, model_prediction(valid_data, clf_svc)))

    model = pickle.load(open(base_path + "model.pk", "rb"))

    mini_data = valid_data[:3]
    predict_stacked_model(mini_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    predict_stacked_model(mini_data, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)

    entry_list = []
    for i in range(len(valid_data)):
        new_entry = valid_data[i:i+1]
        entry_list.append(new_entry)

    preds_y = []
    times = []
    for entry in tqdm(entry_list):
        t0 = time.time()
        pred = predict_stacked_model(entry, model, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
        time_elapsed = time.time() - t0
        times.append(time_elapsed)
        preds_y.append(pred)

    preds_y = np.hstack(preds_y)
    score = willump_score_function(valid_y, preds_y)

    times = np.array(times) * 1000000

    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)
    print("p50 Latency: %f us p99 Latency: %f us" %
          (p50, p99))
    pickle.dump(times, open("latencies.pk", "wb"))