import numpy as np
import pandas as pd
import pickle
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from willump.evaluation.willump_executor import willump_execute

from instant_gratification_utils import *

base_path = "tests/test_resources/instant_gratification/"

training_cascades = {}


@willump_execute(training_cascades=training_cascades, willump_train_function=willump_train_function,
                 willump_predict_function=willump_predict_function, willump_score_function=willump_score_function)
def train_stacked_model(X, y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc):
    pred_svnu = model_prediction(X, clf_svnu)
    pred_knn = model_prediction(X, clf_knn)
    pred_lr = model_prediction(X, clf_lr)
    pred_mlp = model_prediction(X, clf_mlp)
    pred_svc = model_prediction(X, clf_svc)
    combined_pred = np.hstack([pred_svnu, pred_knn, pred_lr, pred_mlp, pred_svc])
    model = willump_train_function(combined_pred, y)
    return model


if __name__ == "__main__":
    data = pd.read_csv(base_path + "train.csv")
    data = data[data['wheezy-copper-turtle-magic'] < NUM_PARTITIONS].reset_index(drop=True)
    print("Data Length: %d" % len(data))
    train_data, _ = train_test_split(data, test_size=0.5, random_state=42)
    train_y = train_data.pop('target')
    del data

    cols = [c for c in train_data.columns if c not in ['id', 'wheezy-copper-turtle-magic']]

    pca_scaler = PCA(svd_solver='full', n_components='mle')
    standard_scaler = StandardScaler()
    scaled_data = pca_scaler.fit_transform(train_data[cols])
    scaled_data = standard_scaler.fit_transform(scaled_data)

    train_models_data, train_stack_data, train_models_y, train_stack_y = train_test_split(scaled_data, train_y,
                                                                                          test_size=0.5,
                                                                                          random_state=42)

    clf_svnu = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)
    clf_svnu.fit(train_models_data, train_models_y)

    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)
    clf_knn.fit(train_models_data, train_models_y)

    clf_lr = linear_model.LogisticRegression(solver='saga', penalty='l1', C=0.1)
    clf_lr.fit(train_models_data, train_models_y)

    clf_mlp = neural_network.MLPClassifier(random_state=3, activation='relu', solver='lbfgs', tol=1e-06,
                                           hidden_layer_sizes=(250,))
    clf_mlp.fit(train_models_data, train_models_y)

    clf_svc = svm.SVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=42)
    clf_svc.fit(train_models_data, train_models_y)

    stacked_model = train_stacked_model(train_stack_data, train_stack_y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    print("First (Python) Train")
    train_stacked_model(train_stack_data, train_stack_y, clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc)
    print("Second (Willump) Train")

    pickle.dump((pca_scaler, standard_scaler), open(base_path + "scaler.pk", "wb"))
    pickle.dump((clf_svnu, clf_knn, clf_lr, clf_mlp, clf_svc), open(base_path + "clf.pk", "wb"))
    pickle.dump(stacked_model, open(base_path + "model.pk", "wb"))
    pickle.dump(training_cascades, open(base_path + "cascades.pk", "wb"))
