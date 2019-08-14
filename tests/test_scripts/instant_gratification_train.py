import argparse
import pickle

import pandas as pd
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from tqdm import tqdm

from instant_gratification_utils import *
from willump.evaluation.willump_executor import willump_execute

base_path = "tests/test_resources/instant_gratification/"

training_cascades = {}

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return")
args = parser.parse_args()


@willump_execute(training_cascades=training_cascades, willump_train_function=willump_train_function,
                 willump_predict_function=willump_predict_function,
                 willump_predict_proba_function=willump_predict_proba_function,
                 willump_score_function=willump_score_function,
                 top_k=args.top_k)
def train_stacked_model(X, y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc):
    pred_svnu = model_prediction(X, clf_svnu)
    # pred_knn = model_prediction(X, clf_knn)
    pred_lr = model_prediction(X, clf_lr)
    pred_mlp = model_prediction(X, clf_mlp)
    pred_svc = model_prediction(X, clf_svc)
    combined_pred = np.hstack([pred_svnu, pred_lr, pred_mlp, pred_svc])
    model = willump_train_function(combined_pred, y)
    return model


if __name__ == "__main__":
    data = pd.read_csv(base_path + "train.csv")
    train_data, _ = train_test_split(data, test_size=0.1, random_state=42)
    train_data = train_data[train_data['wheezy-copper-turtle-magic'] < NUM_PARTITIONS].reset_index(drop=True)
    print("Data Length: %d" % len(data))
    train_y = train_data.pop('target')
    partition_column = train_data.pop('wheezy-copper-turtle-magic').values.reshape(-1, 1)
    del data

    cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

    pca_scaler = PCA(svd_solver='full', n_components='mle')
    standard_scaler = StandardScaler()
    scaled_data = pca_scaler.fit_transform(train_data[cols])
    scaled_data = standard_scaler.fit_transform(scaled_data)
    scaled_data = np.append(scaled_data, partition_column, 1)

    train_models_data, train_stack_data, train_models_y, train_stack_y = train_test_split(scaled_data, train_y,
                                                                                          test_size=0.1,
                                                                                          random_state=42)
    train_models_y = train_models_y.reset_index(drop=True)

    clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc = [], [], [], [], []
    for _ in range(NUM_PARTITIONS):
        clf_svnu.append(NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59,
                              coef0=0.053))
        clf_knn.append(neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9))
        clf_lr.append(linear_model.LogisticRegression(solver='saga', penalty='l1', C=0.1))
        clf_mlp.append(neural_network.MLPClassifier(random_state=3, activation='relu', solver='lbfgs', tol=1e-06,
                                                    hidden_layer_sizes=(250,)))
        clf_svc.append(svm.SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42))

    for i in tqdm(range(NUM_PARTITIONS)):
        partition_indices = np.nonzero(train_models_data[:, PARTITION_INDEX] == i)[0]
        partition_X = train_models_data[partition_indices, :-1]
        partition_y = train_models_y[partition_indices]
        clf_svnu[i].fit(partition_X, partition_y)
        # clf_knn[i].fit(partition_X, partition_y)
        clf_lr[i].fit(partition_X, partition_y)
        clf_mlp[i].fit(partition_X, partition_y)
        clf_svc[i].fit(partition_X, partition_y)

    print(roc_auc_score(train_stack_y, model_prediction(train_stack_data, clf_svnu)))

    stacked_model = train_stacked_model(train_stack_data, train_stack_y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    print("First (Python) Train")
    train_stacked_model(train_stack_data, train_stack_y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    print("Second (Willump) Train")

    pickle.dump((pca_scaler, standard_scaler), open(base_path + "scaler.pk", "wb"))
    pickle.dump((clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc), open(base_path + "clf.pk", "wb"))
    pickle.dump(stacked_model, open(base_path + "model.pk", "wb"))
    pickle.dump(training_cascades, open(base_path + "cascades.pk", "wb"))
