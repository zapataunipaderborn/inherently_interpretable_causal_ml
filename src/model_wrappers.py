import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from pysr import PySRRegressor
from pygam import LinearGAM, s
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# --- Symbolic Regression Wrapper ---
# ---------------------------------------------------------------------
class SymbolicRegressor:
    def __init__(
        self,
        random_state=None,
        maxsize=20,
        niterations=100,
        binary_operators=None,
        unary_operators=None,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        verbosity=0,
        progress=False,
        deterministic=True,
        parallelism='serial',
    ):
        self.random_state = random_state
        self.maxsize = maxsize
        self.niterations = niterations
        self.binary_operators = binary_operators if binary_operators is not None else ["+", "*", "-", "/"]
        self.unary_operators = unary_operators if unary_operators is not None else ["sin", "log", "exp", "log1p"]
        self.elementwise_loss = elementwise_loss
        self.verbosity = verbosity
        self.progress = progress
        self.deterministic = deterministic
        self.parallelism = parallelism
        self.model = PySRRegressor(
            maxsize=self.maxsize,
            niterations=self.niterations,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            elementwise_loss=self.elementwise_loss,
            random_state=self.random_state if self.random_state else 42,
            verbosity=self.verbosity,
            progress=self.progress,
            deterministic=self.deterministic,
            parallelism=self.parallelism,
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def best_equation(self):
        """Return best equation string if available."""
        try:
            if hasattr(self.model, "get_best"):
                best = self.model.get_best()
                if isinstance(best, dict) and "equation" in best:
                    return str(best["equation"])
            return str(self.model)
        except Exception:
            return "No equation available"

# ---------------------------------------------------------------------
# --- GAM Wrapper ---
# ---------------------------------------------------------------------
class GAMRegressorWrapper:
    def __init__(self, random_state=None, max_iter=5000, lam=0.6, n_splines=20, spline_order=3):
        self.random_state = random_state
        self.max_iter = max_iter
        self.lam = lam
        self.n_splines = n_splines
        self.spline_order = spline_order
        self.model = None
        self.n_features = None

    # Required by scikit-learn GridSearchCV / clone()
    def get_params(self, deep=True):
        return {
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "lam": self.lam,
            "n_splines": self.n_splines,
            "spline_order": self.spline_order,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.n_features = X.shape[1]

        formula_terms = s(0, lam=self.lam, n_splines=self.n_splines, spline_order=self.spline_order)
        for i in range(1, self.n_features):
            formula_terms = formula_terms + s(i, lam=self.lam, n_splines=self.n_splines, spline_order=self.spline_order)

        self.model = LinearGAM(formula_terms, max_iter=self.max_iter)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def get_summary(self):
        if self.model is None:
            return "Model not fitted"
        return str(self.model.statistics_)

# ---------------------------------------------------------------------
# --- XGBoost Wrapper ---
# ---------------------------------------------------------------------
class XGBRegressorWrapper:
    def __init__(self, random_state=None, **xgb_params):
        """Wraps XGBRegressor and keeps params visible to GridSearchCV."""
        self.random_state = random_state if random_state is not None else 42
        self.xgb_params = xgb_params
        self.model = XGBRegressor(random_state=self.random_state, verbosity=0, **self.xgb_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importances(self):
        """Return feature importances."""
        return self.model.feature_importances_

    def get_params(self, deep=True):
        """Make wrapper compatible with GridSearchCV."""
        # include both wrapper and underlying XGB params
        params = {"random_state": self.random_state}
        params.update(self.xgb_params)
        return params

    def set_params(self, **params):
        """Update hyperparameters and rebuild the inner model."""
        if "random_state" in params:
            self.random_state = params.pop("random_state")
        self.xgb_params.update(params)
        self.model = XGBRegressor(random_state=self.random_state, verbosity=0, **self.xgb_params)
        return self

# ---------------------------------------------------------------------
# --- MLP Wrapper ---
# ---------------------------------------------------------------------
class MLPRegressorWrapper:
    def __init__(
        self,
        random_state=None,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        learning_rate='constant',
        max_iter=1000,
    ):
        # store hyperparams as attributes so get_params/set_params can see them
        self.random_state = 42 if random_state is None else random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._build_model()

    def _build_model(self):
        # (re)create the inner sklearn estimator from current attributes
        self.model = MLPRegressor(
            random_state=self.random_state,
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
        )

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        # ensure model matches any params changed via set_params
        self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def get_params(self, deep=True):
        # keys must match names used in your param_grid
        return {
            "random_state": self.random_state,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
        }

    def set_params(self, **params):
        # update stored hyperparameters and rebuild inner estimator
        for key, val in params.items():
            if hasattr(self, key):
                setattr(self, key, val)
        self._build_model()
        return self
