import model_wrappers as mw
from sklearn.linear_model import LinearRegression


model_configs = {

    "RandomForest": {
    "model_class": mw.RandomForestRegressor,
    "params": {"random_state": 42},
    "param_grid": {
        "n_estimators": [100, 200, 1000],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 15],
    },
    },
    "LinearRegression": {"model_class": LinearRegression},
    "MLP": {
        "model_class": mw.MLPRegressorWrapper,
        "param_grid": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "alpha": [0.001, 0.01],
        "learning_rate": ['constant', 'adaptive'],
    }},
    "XGBoost": {
        "model_class": mw.XGBRegressorWrapper,
        "params": {"random_state": 42},
        "param_grid": {
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.5],
            "n_estimators": [100, 200, 1000],
        }},
    "GAM": {
        "model_class": mw.GAMRegressorWrapper,
        "params": {},
    },
    "SymbolicRegression": {"model_class": mw.SymbolicRegressor},

    }