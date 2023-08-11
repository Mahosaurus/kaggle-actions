import pandas as pd
import sklearn.linear_model as sc_linear_model

def get_baseline_regression_model(X, Y, COLUMNS):
    model = sc_linear_model.LogisticRegression(penalty='l2', max_iter=1000, solver="liblinear")
    X = X[COLUMNS]
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    model.fit(X, Y)
    return model