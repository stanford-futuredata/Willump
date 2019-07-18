from sklearn.linear_model import LogisticRegression


def willump_train_function(X, y):
    model = LogisticRegression()
    model = model.fit(X, y)
    return model


def willump_predict_function(model, X):
    return model.predict(X)


def willump_predict_proba_function(model, X):
    return model.predict_proba(X)[:, 1]
