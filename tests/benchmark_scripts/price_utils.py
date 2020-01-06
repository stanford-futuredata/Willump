import keras as ks
import numpy as np
from sklearn.metrics import mean_squared_log_error

y_scaler = None


def willump_train_function(X, y):
    model_in = ks.Input(shape=(X.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(192, activation='relu')(model_in)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    model.fit(X, y, batch_size=2 ** (11 + 1), epochs=1, verbose=0)
    return model


def willump_predict_function(model, X):
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    else:
        y_pred = model.predict(X)
        y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
        y_pred[y_pred < 0] = 0
        return y_pred


def willump_predict_proba_function(model, X):
    return willump_predict_function(model, X)


def willump_score_function(true_y, pred_y):
    """
    Expects a scaled true_y but an unscaled pred_y
    """
    true_y = np.expm1(y_scaler.inverse_transform(true_y.reshape(-1, 1))[:, 0])
    return 1 - np.sqrt(mean_squared_log_error(true_y, pred_y))
