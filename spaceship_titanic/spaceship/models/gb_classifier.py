import sklearn.ensemble as sc_ensemble

def get_gb_classifier(X, Y, COLUMNS):
    model = sc_ensemble.RandomForestClassifier(n_estimators=100, max_features=4, max_depth=3)
    X = X[COLUMNS]
    model.fit(X, Y)
    return model