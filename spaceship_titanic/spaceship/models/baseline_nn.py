import sklearn.neural_network as sc_neural_network

def get_baseline_nn_model(X, Y, COLUMNS):
    model = sc_neural_network.MLPClassifier(hidden_layer_sizes=(30, 30),)
    X = X[COLUMNS]
    model.fit(X, Y)
    return model