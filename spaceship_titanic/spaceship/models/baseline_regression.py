import pandas as pd
import sklearn.linear_model as sc_linear_model

def get_baseline_regression_model(X, Y, COLUMNS):
    model = sc_linear_model.LogisticRegression()
    X = X[COLUMNS]
    if isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)
    model.fit(X, Y)
    return model